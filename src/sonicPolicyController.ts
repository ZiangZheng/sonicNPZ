import * as ort from 'onnxruntime-web';
import type loadMujoco from '@mujoco/mujoco';

type MujocoModule = Awaited<ReturnType<typeof loadMujoco>>;

const G1_DOF = 29;
const WBC_DOF = 15;
const CONTROL_DT = 0.02;

const ISAACLAB_TO_MUJOCO = [
  0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8,
  11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28,
];

const MUJOCO_TO_ISAACLAB = [
  0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10,
  16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28,
];

const DEFAULT_ANGLES = [
  -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
  -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
  0.0, 0.0, 0.0,
  0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
  0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
];

const ARMATURE_5020 = 0.003609725;
const ARMATURE_7520_14 = 0.010177520;
const ARMATURE_7520_22 = 0.025101925;
const ARMATURE_4010 = 0.00425;
const NATURAL_FREQ = 10 * 2.0 * Math.PI;
const DAMPING_RATIO = 2;

const stiffness = (armature: number) => armature * NATURAL_FREQ * NATURAL_FREQ;
const damping = (armature: number) => 2.0 * DAMPING_RATIO * armature * NATURAL_FREQ;

const STIFFNESS_5020 = stiffness(ARMATURE_5020);
const STIFFNESS_7520_14 = stiffness(ARMATURE_7520_14);
const STIFFNESS_7520_22 = stiffness(ARMATURE_7520_22);
const STIFFNESS_4010 = stiffness(ARMATURE_4010);

const DAMPING_5020 = damping(ARMATURE_5020);
const DAMPING_7520_14 = damping(ARMATURE_7520_14);
const DAMPING_7520_22 = damping(ARMATURE_7520_22);
const DAMPING_4010 = damping(ARMATURE_4010);

const EFFORT_LIMIT_5020 = 25.0;
const EFFORT_LIMIT_7520_14 = 88.0;
const EFFORT_LIMIT_7520_22 = 139.0;
const EFFORT_LIMIT_4010 = 5.0;

const KPS = [
  STIFFNESS_7520_22, STIFFNESS_7520_22, STIFFNESS_7520_14, STIFFNESS_7520_22, 2 * STIFFNESS_5020, 2 * STIFFNESS_5020,
  STIFFNESS_7520_22, STIFFNESS_7520_22, STIFFNESS_7520_14, STIFFNESS_7520_22, 2 * STIFFNESS_5020, 2 * STIFFNESS_5020,
  STIFFNESS_7520_14, 2 * STIFFNESS_5020, 2 * STIFFNESS_5020,
  STIFFNESS_5020, STIFFNESS_5020, STIFFNESS_5020, STIFFNESS_5020, STIFFNESS_5020, STIFFNESS_4010, STIFFNESS_4010,
  STIFFNESS_5020, STIFFNESS_5020, STIFFNESS_5020, STIFFNESS_5020, STIFFNESS_5020, STIFFNESS_4010, STIFFNESS_4010,
];

const KDS = [
  DAMPING_7520_22, DAMPING_7520_22, DAMPING_7520_14, DAMPING_7520_22, 2 * DAMPING_5020, 2 * DAMPING_5020,
  DAMPING_7520_22, DAMPING_7520_22, DAMPING_7520_14, DAMPING_7520_22, 2 * DAMPING_5020, 2 * DAMPING_5020,
  DAMPING_7520_14, 2 * DAMPING_5020, 2 * DAMPING_5020,
  DAMPING_5020, DAMPING_5020, DAMPING_5020, DAMPING_5020, DAMPING_5020, DAMPING_4010, DAMPING_4010,
  DAMPING_5020, DAMPING_5020, DAMPING_5020, DAMPING_5020, DAMPING_5020, DAMPING_4010, DAMPING_4010,
];

const ACTION_SCALE = [
  0.25 * EFFORT_LIMIT_7520_22 / STIFFNESS_7520_22,
  0.25 * EFFORT_LIMIT_7520_22 / STIFFNESS_7520_22,
  0.25 * EFFORT_LIMIT_7520_14 / STIFFNESS_7520_14,
  0.25 * EFFORT_LIMIT_7520_22 / STIFFNESS_7520_22,
  0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020,
  0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020,
  0.25 * EFFORT_LIMIT_7520_22 / STIFFNESS_7520_22,
  0.25 * EFFORT_LIMIT_7520_22 / STIFFNESS_7520_22,
  0.25 * EFFORT_LIMIT_7520_14 / STIFFNESS_7520_14,
  0.25 * EFFORT_LIMIT_7520_22 / STIFFNESS_7520_22,
  0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020,
  0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020,
  0.25 * EFFORT_LIMIT_7520_14 / STIFFNESS_7520_14,
  0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020,
  0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020,
  0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020,
  0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020,
  0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020,
  0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020,
  0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020,
  0.25 * EFFORT_LIMIT_4010 / STIFFNESS_4010,
  0.25 * EFFORT_LIMIT_4010 / STIFFNESS_4010,
  0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020,
  0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020,
  0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020,
  0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020,
  0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020,
  0.25 * EFFORT_LIMIT_4010 / STIFFNESS_4010,
  0.25 * EFFORT_LIMIT_4010 / STIFFNESS_4010,
];

const TORQUE_LIMIT = [
  139, 139, 88, 139, 50, 50,
  139, 139, 88, 139, 50, 50,
  88, 50, 50,
  25, 25, 25, 25, 25, 5, 5,
  25, 25, 25, 25, 25, 5, 5,
];

export interface SonicPolicySummary {
  encoderInputs: string[];
  encoderOutputs: string[];
  decoderInputs: string[];
  decoderOutputs: string[];
  decoderInputSize: number;
  decoderOutputSize: number;
  balanceInputs: string[];
  balanceOutputs: string[];
}

interface HistoryEntry {
  baseQuat: number[];
  baseAngVel: number[];
  bodyQ: number[];
  bodyDq: number[];
  lastAction: number[];
}

export class SonicPolicyController {
  private mujoco: MujocoModule;
  private encoder?: ort.InferenceSession;
  private decoder?: ort.InferenceSession;
  private balance?: ort.InferenceSession;
  private token = new Float32Array(64);
  private action = new Float32Array(G1_DOF);
  private targetQ = Float32Array.from(DEFAULT_ANGLES);
  private balanceAction = new Float32Array(WBC_DOF);
  private balanceTargetQ = Float32Array.from([-0.1, 0.0, 0.0, 0.3, -0.2, 0.0, -0.1, 0.0, 0.0, 0.3, -0.2, 0.0, 0.0, 0.0, 0.0]);
  private balanceObs = new Float32Array(516);
  private balanceHistory: Float32Array[] = [];
  private obs = new Float32Array(994);
  private history: HistoryEntry[] = [];
  private stepCounter = 0;
  private decimation = 10;

  constructor(mujoco: MujocoModule) {
    this.mujoco = mujoco;
  }

  async loadFromURLs(encoderUrl: string, decoderUrl: string, balanceUrl: string): Promise<SonicPolicySummary> {
    ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';
    const [encoderBytes, decoderBytes, balanceBytes] = await Promise.all([
      fetchArrayBuffer(encoderUrl),
      fetchArrayBuffer(decoderUrl),
      fetchArrayBuffer(balanceUrl),
    ]);
    return this.loadFromBuffers(encoderBytes, decoderBytes, balanceBytes);
  }

  async loadFromFiles(files: FileList | File[]): Promise<SonicPolicySummary> {
    const allFiles = Array.from(files);
    const encoderFile = allFiles.find((file) => /encoder/i.test(file.name));
    const decoderFile = allFiles.find((file) => /decoder/i.test(file.name));
    if (!encoderFile || !decoderFile) {
      throw new Error('Select both Sonic encoder and decoder ONNX files.');
    }
    return this.loadFromBuffers(await encoderFile.arrayBuffer(), await decoderFile.arrayBuffer());
  }

  reset(model: any, data: any): void {
    this.stepCounter = 0;
    this.action.fill(0);
    this.balanceAction.fill(0);
    this.token.fill(0);
    this.targetQ = Float32Array.from(DEFAULT_ANGLES);
    this.balanceTargetQ = Float32Array.from([-0.1, 0.0, 0.0, 0.3, -0.2, 0.0, -0.1, 0.0, 0.0, 0.3, -0.2, 0.0, 0.0, 0.0, 0.0]);
    this.decimation = Math.max(1, Math.round(CONTROL_DT / Number(model.opt.timestep || 0.002)));
    this.history = [];
    this.balanceHistory = [];
    for (let i = 0; i < 10; i += 1) {
      this.history.push(this.readHistoryEntry(data));
    }
    for (let i = 0; i < 6; i += 1) {
      this.balanceHistory.push(this.buildBalanceSingleObservation(data));
    }
  }

  setInitialStandingState(model: any, data: any): void {
    this.mujoco.mj_resetData(model, data);
    data.qpos[0] = 0;
    data.qpos[1] = 0;
    data.qpos[2] = 0.74;
    data.qpos[3] = 1;
    data.qpos[4] = 0;
    data.qpos[5] = 0;
    data.qpos[6] = 0;
    for (let i = 0; i < G1_DOF && 7 + i < model.nq; i += 1) {
      data.qpos[7 + i] = DEFAULT_ANGLES[i];
    }
    for (let i = 0; i < model.nv; i += 1) data.qvel[i] = 0;
    this.mujoco.mj_forward(model, data);
    this.reset(model, data);
  }

  async maybeUpdateAction(model: any, data: any): Promise<void> {
    if (this.stepCounter % this.decimation !== 0) return;
    if (this.balance) {
      this.balanceHistory.push(this.buildBalanceSingleObservation(data));
      while (this.balanceHistory.length > 6) this.balanceHistory.shift();
      this.balanceObs.fill(0);
      for (let i = 0; i < this.balanceHistory.length; i += 1) {
        this.balanceObs.set(this.balanceHistory[i], i * 86);
      }
      const inputName = this.balance.inputNames[0];
      const outputName = this.balance.outputNames[0];
      const result = await this.balance.run({
        [inputName]: new ort.Tensor('float32', this.balanceObs, [1, this.balanceObs.length]),
      });
      const output = result[outputName].data as Float32Array;
      for (let i = 0; i < WBC_DOF && i < output.length; i += 1) {
        this.balanceAction[i] = output[i];
        this.balanceTargetQ[i] = this.balanceDefault(i) + this.balanceAction[i] * 0.25;
      }
      return;
    }
    if (!this.decoder) return;
    this.pushHistory(data);
    this.buildDecoderObservation();
    const inputName = this.decoder.inputNames[0];
    const outputName = this.decoder.outputNames[0];
    const result = await this.decoder.run({
      [inputName]: new ort.Tensor('float32', this.obs, [1, this.obs.length]),
    });
    const output = result[outputName].data as Float32Array;
    if (output.length < G1_DOF) return;
    for (let policyIndex = 0; policyIndex < G1_DOF; policyIndex += 1) {
      this.action[policyIndex] = output[policyIndex];
    }
    for (let mujocoIndex = 0; mujocoIndex < G1_DOF; mujocoIndex += 1) {
      const policyIndex = ISAACLAB_TO_MUJOCO[mujocoIndex];
      this.targetQ[mujocoIndex] = DEFAULT_ANGLES[mujocoIndex] + this.action[policyIndex] * ACTION_SCALE[mujocoIndex];
    }
  }

  applyControl(model: any, data: any): void {
    for (let i = 0; i < model.nu; i += 1) data.ctrl[i] = 0;
    for (let i = 0; i < model.nv; i += 1) data.qfrc_applied[i] = 0;

    if (this.balance) {
      const balanceKp = [150, 150, 150, 200, 40, 40, 150, 150, 150, 200, 40, 40, 250, 250, 250];
      const balanceKd = [2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2, 5, 5, 5];
      for (let i = 0; i < WBC_DOF; i += 1) {
        const torque = balanceKp[i] * (this.balanceTargetQ[i] - Number(data.qpos[7 + i] ?? 0))
          - balanceKd[i] * Number(data.qvel[6 + i] ?? 0);
        data.ctrl[i] = clamp(torque, -TORQUE_LIMIT[i], TORQUE_LIMIT[i]);
      }
      for (let i = WBC_DOF; i < G1_DOF; i += 1) {
        const torque = -100 * Number(data.qpos[7 + i] ?? 0) - 0.5 * Number(data.qvel[6 + i] ?? 0);
        data.ctrl[i] = clamp(torque, -TORQUE_LIMIT[i], TORQUE_LIMIT[i]);
      }
      this.stepCounter += 1;
      return;
    }

    for (let mujocoIndex = 0; mujocoIndex < G1_DOF; mujocoIndex += 1) {
      const qposAdr = 7 + mujocoIndex;
      const qvelAdr = 6 + mujocoIndex;
      if (qposAdr >= model.nq || qvelAdr >= model.nv || mujocoIndex >= model.nu) continue;
      const torque = KPS[mujocoIndex] * (this.targetQ[mujocoIndex] - data.qpos[qposAdr])
        - KDS[mujocoIndex] * data.qvel[qvelAdr];
      data.ctrl[mujocoIndex] = clamp(torque, -TORQUE_LIMIT[mujocoIndex], TORQUE_LIMIT[mujocoIndex]);
    }
    this.stepCounter += 1;
  }

  isReady(): boolean {
    return Boolean(this.balance || this.decoder);
  }

  private async loadFromBuffers(
    encoderBytes: ArrayBuffer,
    decoderBytes: ArrayBuffer,
    balanceBytes?: ArrayBuffer,
  ): Promise<SonicPolicySummary> {
    this.encoder = await ort.InferenceSession.create(encoderBytes, { executionProviders: ['wasm'] });
    this.decoder = await ort.InferenceSession.create(decoderBytes, { executionProviders: ['wasm'] });
    if (balanceBytes) {
      this.balance = await ort.InferenceSession.create(balanceBytes, { executionProviders: ['wasm'] });
    }
    return {
      encoderInputs: [...this.encoder.inputNames],
      encoderOutputs: [...this.encoder.outputNames],
      decoderInputs: [...this.decoder.inputNames],
      decoderOutputs: [...this.decoder.outputNames],
      decoderInputSize: this.obs.length,
      decoderOutputSize: G1_DOF,
      balanceInputs: this.balance ? [...this.balance.inputNames] : [],
      balanceOutputs: this.balance ? [...this.balance.outputNames] : [],
    };
  }

  private pushHistory(data: any) {
    this.history.push(this.readHistoryEntry(data));
    while (this.history.length > 10) this.history.shift();
  }

  private readHistoryEntry(data: any): HistoryEntry {
    const baseQuat = [
      Number(data.qpos[3] ?? 1),
      Number(data.qpos[4] ?? 0),
      Number(data.qpos[5] ?? 0),
      Number(data.qpos[6] ?? 0),
    ];
    const bodyQ = new Array<number>(G1_DOF);
    const bodyDq = new Array<number>(G1_DOF);
    for (let policyIndex = 0; policyIndex < G1_DOF; policyIndex += 1) {
      const mujocoIndex = MUJOCO_TO_ISAACLAB[policyIndex];
      bodyQ[policyIndex] = Number(data.qpos[7 + mujocoIndex] ?? 0) - DEFAULT_ANGLES[mujocoIndex];
      bodyDq[policyIndex] = Number(data.qvel[6 + mujocoIndex] ?? 0);
    }
    return {
      baseQuat,
      baseAngVel: [Number(data.qvel[3] ?? 0), Number(data.qvel[4] ?? 0), Number(data.qvel[5] ?? 0)],
      bodyQ,
      bodyDq,
      lastAction: Array.from(this.action),
    };
  }

  private buildDecoderObservation(): void {
    this.obs.fill(0);
    let offset = 0;
    this.obs.set(this.token, offset);
    offset += 64;
    for (const entry of this.latestHistory()) {
      this.obs.set(entry.baseAngVel, offset);
      offset += 3;
    }
    for (const entry of this.latestHistory()) {
      this.obs.set(entry.bodyQ, offset);
      offset += G1_DOF;
    }
    for (const entry of this.latestHistory()) {
      this.obs.set(entry.bodyDq, offset);
      offset += G1_DOF;
    }
    for (const entry of this.latestHistory()) {
      this.obs.set(entry.lastAction, offset);
      offset += G1_DOF;
    }
    for (const entry of this.latestHistory()) {
      this.obs.set(projectGravity(entry.baseQuat), offset);
      offset += 3;
    }
  }

  private buildBalanceSingleObservation(data: any): Float32Array {
    const single = new Float32Array(86);
    single[3] = 0.74;
    single.set([
      Number(data.qvel[3] ?? 0) * 0.5,
      Number(data.qvel[4] ?? 0) * 0.5,
      Number(data.qvel[5] ?? 0) * 0.5,
    ], 7);
    single.set(projectGravity([
      Number(data.qpos[3] ?? 1),
      Number(data.qpos[4] ?? 0),
      Number(data.qpos[5] ?? 0),
      Number(data.qpos[6] ?? 0),
    ]), 10);
    for (let i = 0; i < G1_DOF; i += 1) {
      const defaultAngle = i < WBC_DOF ? this.balanceDefault(i) : 0;
      single[13 + i] = Number(data.qpos[7 + i] ?? 0) - defaultAngle;
      single[13 + G1_DOF + i] = Number(data.qvel[6 + i] ?? 0) * 0.05;
    }
    single.set(this.balanceAction, 13 + 2 * G1_DOF);
    return single;
  }

  private balanceDefault(index: number): number {
    return [-0.1, 0.0, 0.0, 0.3, -0.2, 0.0, -0.1, 0.0, 0.0, 0.3, -0.2, 0.0, 0.0, 0.0, 0.0][index] ?? 0;
  }

  private latestHistory(): HistoryEntry[] {
    if (this.history.length >= 10) return this.history.slice(-10);
    const first = this.history[0] ?? this.readHistoryEntry({ qpos: [], qvel: [] });
    return [...Array.from({ length: 10 - this.history.length }, () => first), ...this.history];
  }
}

async function fetchArrayBuffer(url: string): Promise<ArrayBuffer> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to fetch Sonic ONNX: ${url} (${res.status})`);
  return res.arrayBuffer();
}

function projectGravity(quat: number[]): number[] {
  const [w, x, y, z] = quat;
  const qConj = [w, -x, -y, -z];
  const v = [0, 0, -1];
  return [
    v[0] * (qConj[0] ** 2 + qConj[1] ** 2 - qConj[2] ** 2 - qConj[3] ** 2)
      + v[1] * 2 * (qConj[1] * qConj[2] - qConj[0] * qConj[3])
      + v[2] * 2 * (qConj[1] * qConj[3] + qConj[0] * qConj[2]),
    v[0] * 2 * (qConj[1] * qConj[2] + qConj[0] * qConj[3])
      + v[1] * (qConj[0] ** 2 - qConj[1] ** 2 + qConj[2] ** 2 - qConj[3] ** 2)
      + v[2] * 2 * (qConj[2] * qConj[3] - qConj[0] * qConj[1]),
    v[0] * 2 * (qConj[1] * qConj[3] - qConj[0] * qConj[2])
      + v[1] * 2 * (qConj[2] * qConj[3] + qConj[0] * qConj[1])
      + v[2] * (qConj[0] ** 2 - qConj[1] ** 2 - qConj[2] ** 2 + qConj[3] ** 2),
  ];
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}
