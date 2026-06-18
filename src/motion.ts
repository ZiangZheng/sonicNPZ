import { unzipSync } from 'fflate';

export interface MotionClip {
  fps: number;
  duration: number;
  jointNames: string[];
  times: Float32Array;
  qpos: Float32Array[];
  qvel: Float32Array[];
  rootPos: Float32Array[];
  rootQuat: Float32Array[];
  leftHandCmd?: Float32Array[];
  rightHandCmd?: Float32Array[];
}

export async function loadMotionFromURL(url: string): Promise<MotionClip> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to load motion: ${url}`);
  if (/\.npz($|\?)/i.test(url)) {
    return normalizeNpz(await res.arrayBuffer());
  }
  const json = await res.json();
  return normalizeMotion(json);
}

export function loadMotionFromFile(file: File): Promise<MotionClip> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      try {
        if (/\.npz$/i.test(file.name)) {
          resolve(normalizeNpz(reader.result as ArrayBuffer));
          return;
        }
        const json = JSON.parse(reader.result as string);
        resolve(normalizeMotion(json));
      } catch (e) {
        reject(e);
      }
    };
    reader.onerror = reject;
    if (/\.npz$/i.test(file.name)) {
      reader.readAsArrayBuffer(file);
    } else {
      reader.readAsText(file);
    }
  });
}

function normalizeMotion(raw: any): MotionClip {
  const fps = raw.fps || 30;
  const times = new Float32Array(raw.times);
  const qpos = (raw.qpos as number[][]).map((row) => new Float32Array(row));
  const qvel = (raw.qvel as number[][]).map((row) => new Float32Array(row));
  const rootPos = (raw.root_pos as number[][]).map((row) => new Float32Array(row));
  const rootQuat = (raw.root_quat as number[][]).map((row) => new Float32Array(row));
  const duration = raw.duration || times[times.length - 1];
  return {
    fps,
    duration,
    jointNames: raw.joint_names || [],
    times,
    qpos,
    qvel,
    rootPos,
    rootQuat,
    leftHandCmd: raw.left_hand_cmd?.map((row: number[]) => new Float32Array(row)),
    rightHandCmd: raw.right_hand_cmd?.map((row: number[]) => new Float32Array(row)),
  };
}

export function sampleMotion(
  motion: MotionClip,
  time: number,
): { qpos: Float32Array; qvel: Float32Array; idx: number; alpha: number; leftHandCmd?: Float32Array; rightHandCmd?: Float32Array } {
  const { times, qpos, qvel } = motion;
  const n = times.length;
  if (n === 0) return { qpos: new Float32Array(0), qvel: new Float32Array(0), idx: 0, alpha: 0 };
  if (n === 1 || time >= times[n - 1]) {
    const idx = n - 1;
    return {
      qpos: new Float32Array(qpos[idx]),
      qvel: new Float32Array(qvel[idx]),
      idx,
      alpha: 0,
      leftHandCmd: cloneOptionalRow(motion.leftHandCmd?.[idx]),
      rightHandCmd: cloneOptionalRow(motion.rightHandCmd?.[idx]),
    };
  }

  const t = Math.max(0, time);
  let idx = 0;
  for (let i = 0; i < n - 1; i++) {
    if (t >= times[i] && t < times[i + 1]) {
      idx = i;
      break;
    }
  }
  const dt = times[idx + 1] - times[idx];
  const alpha = dt > 0 ? (t - times[idx]) / dt : 0;

  const a = qpos[idx];
  const b = qpos[Math.min(idx + 1, n - 1)];
  const outQpos = new Float32Array(a.length);
  outQpos[0] = a[0] + alpha * (b[0] - a[0]);
  outQpos[1] = a[1] + alpha * (b[1] - a[1]);
  outQpos[2] = a[2] + alpha * (b[2] - a[2]);
  // Spherical linear interpolation for root quaternion.
  slerp(a, b, outQpos, 3, alpha);
  for (let i = 7; i < a.length; i++) {
    outQpos[i] = a[i] + alpha * (b[i] - a[i]);
  }

  const va = qvel[idx];
  const vb = qvel[Math.min(idx + 1, n - 1)];
  const outQvel = new Float32Array(va.length);
  for (let i = 0; i < va.length; i++) {
    outQvel[i] = va[i] + alpha * (vb[i] - va[i]);
  }

  const leftHandCmd = interpolateOptionalRows(motion.leftHandCmd, idx, alpha);
  const rightHandCmd = interpolateOptionalRows(motion.rightHandCmd, idx, alpha);

  return { qpos: outQpos, qvel: outQvel, idx, alpha, leftHandCmd, rightHandCmd };
}

function normalizeNpz(buffer: ArrayBuffer): MotionClip {
  const files = unzipSync(new Uint8Array(buffer));
  const arrays = new Map<string, NpyArray>();
  for (const [name, bytes] of Object.entries(files)) {
    if (name.endsWith('.npy')) {
      arrays.set(name.replace(/\.npy$/, ''), parseNpy(bytes));
    }
  }

  if (arrays.has('qpos')) {
    return normalizeMotion({
      fps: scalarNumber(arrays.get('fps')) || 30,
      times: matrixRows(arrays.get('times')),
      qpos: matrixRows(arrays.get('qpos')),
      qvel: matrixRows(arrays.get('qvel')),
      root_pos: matrixRows(arrays.get('root_pos')),
      root_quat: matrixRows(arrays.get('root_quat')),
      joint_names: [],
    });
  }

  return humanIdmNpzToMotion(arrays);
}

function humanIdmNpzToMotion(arrays: Map<string, NpyArray>): MotionClip {
  const prefix = arrays.has('gt_gt_lower_qpos')
    ? 'gt_gt'
    : arrays.has('pred100_pred_lower_qpos')
      ? 'pred100_pred'
      : 'pred50_pred';
  const lower = requireMatrix(arrays, `${prefix}_lower_qpos`);
  const leftArm = requireMatrix(arrays, `${prefix}_left_arm_pose`);
  const rightArm = requireMatrix(arrays, `${prefix}_right_arm_pose`);
  const leftHandCmd = matrixToFloatRows(arrays.get(`${prefix}_left_hand_cmd`));
  const rightHandCmd = matrixToFloatRows(arrays.get(`${prefix}_right_hand_cmd`));

  const n = Math.min(lower.length, leftArm.length, rightArm.length);
  const fps = scalarNumber(arrays.get('fps')) || 30;
  const times = Float32Array.from({ length: n }, (_, i) => i / fps);
  const qpos = Array.from({ length: n }, (_, i) => {
    const row = new Float32Array(36);
    row[0] = 0;
    row[1] = 0;
    row[2] = 0.78;
    row[3] = 1;
    row[4] = 0;
    row[5] = 0;
    row[6] = 0;
    row.set(lower[i].subarray(0, 15), 7);
    row.set(leftArm[i].subarray(0, 7), 7 + 15);
    row.set(rightArm[i].subarray(0, 7), 7 + 22);
    return row;
  });
  const qvel = estimateQvel(qpos, fps);

  return {
    fps,
    duration: n > 0 ? (n - 1) / fps : 0,
    jointNames: [],
    times,
    qpos,
    qvel,
    rootPos: qpos.map((row) => row.slice(0, 3)),
    rootQuat: qpos.map((row) => row.slice(3, 7)),
    leftHandCmd,
    rightHandCmd,
  };
}

interface NpyArray {
  shape: number[];
  data: Float64Array | Float32Array | Int32Array | BigInt64Array | Uint8Array;
}

function parseNpy(bytes: Uint8Array): NpyArray {
  const magic = String.fromCharCode(...bytes.subarray(0, 6));
  if (magic !== '\x93NUMPY') throw new Error('Invalid npy file inside npz');
  const major = bytes[6];
  const headerLenOffset = 8;
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const headerLength = major <= 1
    ? view.getUint16(headerLenOffset, true)
    : view.getUint32(headerLenOffset, true);
  const dataOffset = headerLenOffset + (major <= 1 ? 2 : 4) + headerLength;
  const header = new TextDecoder().decode(bytes.subarray(headerLenOffset + (major <= 1 ? 2 : 4), dataOffset));
  if (/True/.test(header.match(/'fortran_order':\s*(True|False)/)?.[1] ?? 'False')) {
    throw new Error('Fortran-order npy arrays are not supported');
  }
  const descr = header.match(/'descr':\s*'([^']+)'/)?.[1];
  const shapeText = header.match(/'shape':\s*\(([^)]*)\)/)?.[1] ?? '';
  const shape = shapeText.split(',').map((part) => Number(part.trim())).filter(Number.isFinite);
  if (!descr) throw new Error('Missing npy dtype descriptor');
  const body = bytes.buffer.slice(bytes.byteOffset + dataOffset, bytes.byteOffset + bytes.byteLength);
  if (descr.endsWith('f4')) return { shape, data: new Float32Array(body) };
  if (descr.endsWith('f8')) return { shape, data: new Float64Array(body) };
  if (descr.endsWith('i4')) return { shape, data: new Int32Array(body) };
  if (descr.endsWith('i8')) return { shape, data: new BigInt64Array(body) };
  if (descr.endsWith('u1')) return { shape, data: new Uint8Array(body) };
  throw new Error(`Unsupported npy dtype: ${descr}`);
}

function requireMatrix(arrays: Map<string, NpyArray>, name: string): Float32Array[] {
  const rows = matrixToFloatRows(arrays.get(name));
  if (!rows.length) throw new Error(`NPZ missing required array: ${name}`);
  return rows;
}

function matrixToFloatRows(array?: NpyArray): Float32Array[] | undefined {
  if (!array) return undefined;
  const rows = matrixRows(array);
  return rows.map((row) => Float32Array.from(row));
}

function matrixRows(array?: NpyArray): number[][] {
  if (!array) return [];
  const shape = array.shape.length ? array.shape : [array.data.length];
  if (shape.length === 1) return [Array.from(array.data as ArrayLike<number>)];
  const rows = shape[0];
  const cols = shape.slice(1).reduce((a, b) => a * b, 1);
  const out: number[][] = [];
  for (let r = 0; r < rows; r += 1) {
    const row: number[] = [];
    for (let c = 0; c < cols; c += 1) {
      const value = array.data[r * cols + c] as number | bigint;
      row.push(typeof value === 'bigint' ? Number(value) : Number(value));
    }
    out.push(row);
  }
  return out;
}

function scalarNumber(array?: NpyArray): number | undefined {
  if (!array || array.data.length < 1) return undefined;
  const value = array.data[0] as number | bigint;
  return typeof value === 'bigint' ? Number(value) : Number(value);
}

function estimateQvel(qpos: Float32Array[], fps: number): Float32Array[] {
  return qpos.map((row, i) => {
    const prev = qpos[Math.max(0, i - 1)];
    const next = qpos[Math.min(qpos.length - 1, i + 1)];
    const out = new Float32Array(35);
    for (let j = 0; j < 3; j += 1) out[j] = (next[j] - prev[j]) * fps * 0.5;
    for (let j = 0; j < 29; j += 1) out[6 + j] = (next[7 + j] - prev[7 + j]) * fps * 0.5;
    return out;
  });
}

function interpolateOptionalRows(rows: Float32Array[] | undefined, idx: number, alpha: number) {
  if (!rows?.length) return undefined;
  const a = rows[idx];
  const b = rows[Math.min(idx + 1, rows.length - 1)];
  const out = new Float32Array(a.length);
  for (let i = 0; i < a.length; i += 1) out[i] = a[i] + alpha * (b[i] - a[i]);
  return out;
}

function cloneOptionalRow(row?: Float32Array) {
  return row ? new Float32Array(row) : undefined;
}

function slerp(
  a: Float32Array,
  b: Float32Array,
  out: Float32Array,
  offset: number,
  t: number,
) {
  let dot =
    a[offset] * b[offset] +
    a[offset + 1] * b[offset + 1] +
    a[offset + 2] * b[offset + 2] +
    a[offset + 3] * b[offset + 3];
  let qb0 = b[offset];
  let qb1 = b[offset + 1];
  let qb2 = b[offset + 2];
  let qb3 = b[offset + 3];
  if (dot < 0) {
    dot = -dot;
    qb0 = -qb0;
    qb1 = -qb1;
    qb2 = -qb2;
    qb3 = -qb3;
  }
  let theta0, theta, s0, s1;
  if (dot > 0.9995) {
    out[offset] = a[offset] + t * (qb0 - a[offset]);
    out[offset + 1] = a[offset + 1] + t * (qb1 - a[offset + 1]);
    out[offset + 2] = a[offset + 2] + t * (qb2 - a[offset + 2]);
    out[offset + 3] = a[offset + 3] + t * (qb3 - a[offset + 3]);
    const invLen = 1 / Math.hypot(out[offset], out[offset + 1], out[offset + 2], out[offset + 3]);
    out[offset] *= invLen;
    out[offset + 1] *= invLen;
    out[offset + 2] *= invLen;
    out[offset + 3] *= invLen;
    return;
  }
  theta0 = Math.acos(dot);
  theta = theta0 * t;
  s0 = Math.cos(theta) - dot * Math.sin(theta) / Math.sin(theta0);
  s1 = Math.sin(theta) / Math.sin(theta0);
  out[offset] = a[offset] * s0 + qb0 * s1;
  out[offset + 1] = a[offset + 1] * s0 + qb1 * s1;
  out[offset + 2] = a[offset + 2] * s0 + qb2 * s1;
  out[offset + 3] = a[offset + 3] * s0 + qb3 * s1;
}
