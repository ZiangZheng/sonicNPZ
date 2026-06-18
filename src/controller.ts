import type loadMujoco from '@mujoco/mujoco';

type MujocoModule = Awaited<ReturnType<typeof loadMujoco>>;

export interface ControllerOptions {
  kp: number;
  kd: number;
  maxTorque: number;
  rootAssist: boolean;
}

export const DEFAULT_CONTROLLER_OPTIONS: ControllerOptions = {
  kp: 1,
  kd: 1,
  maxTorque: 1,
  rootAssist: false,
};

const jointToActuatorCache = new WeakMap<object, Map<number, number>>();
const jointNameCache = new WeakMap<object, string[]>();

const WBC_KP: Record<string, number> = {
  left_hip_pitch_joint: 150,
  left_hip_roll_joint: 150,
  left_hip_yaw_joint: 150,
  left_knee_joint: 200,
  left_ankle_pitch_joint: 40,
  left_ankle_roll_joint: 40,
  right_hip_pitch_joint: 150,
  right_hip_roll_joint: 150,
  right_hip_yaw_joint: 150,
  right_knee_joint: 200,
  right_ankle_pitch_joint: 40,
  right_ankle_roll_joint: 40,
  waist_yaw_joint: 250,
  waist_roll_joint: 250,
  waist_pitch_joint: 250,
  left_shoulder_pitch_joint: 100,
  left_shoulder_roll_joint: 100,
  left_shoulder_yaw_joint: 40,
  left_elbow_joint: 40,
  left_wrist_roll_joint: 20,
  left_wrist_pitch_joint: 20,
  left_wrist_yaw_joint: 20,
  right_shoulder_pitch_joint: 100,
  right_shoulder_roll_joint: 100,
  right_shoulder_yaw_joint: 40,
  right_elbow_joint: 40,
  right_wrist_roll_joint: 20,
  right_wrist_pitch_joint: 20,
  right_wrist_yaw_joint: 20,
};

const WBC_KD: Record<string, number> = {
  left_hip_pitch_joint: 2,
  left_hip_roll_joint: 2,
  left_hip_yaw_joint: 2,
  left_knee_joint: 4,
  left_ankle_pitch_joint: 2,
  left_ankle_roll_joint: 2,
  right_hip_pitch_joint: 2,
  right_hip_roll_joint: 2,
  right_hip_yaw_joint: 2,
  right_knee_joint: 4,
  right_ankle_pitch_joint: 2,
  right_ankle_roll_joint: 2,
  waist_yaw_joint: 5,
  waist_roll_joint: 5,
  waist_pitch_joint: 5,
  left_shoulder_pitch_joint: 5,
  left_shoulder_roll_joint: 5,
  left_shoulder_yaw_joint: 2,
  left_elbow_joint: 2,
  left_wrist_roll_joint: 2,
  left_wrist_pitch_joint: 2,
  left_wrist_yaw_joint: 2,
  right_shoulder_pitch_joint: 5,
  right_shoulder_roll_joint: 5,
  right_shoulder_yaw_joint: 2,
  right_elbow_joint: 2,
  right_wrist_roll_joint: 2,
  right_wrist_pitch_joint: 2,
  right_wrist_yaw_joint: 2,
};

const TORQUE_LIMIT: Record<string, number> = {
  left_hip_pitch_joint: 88,
  left_hip_roll_joint: 139,
  left_hip_yaw_joint: 88,
  left_knee_joint: 139,
  left_ankle_pitch_joint: 50,
  left_ankle_roll_joint: 50,
  right_hip_pitch_joint: 88,
  right_hip_roll_joint: 139,
  right_hip_yaw_joint: 88,
  right_knee_joint: 139,
  right_ankle_pitch_joint: 50,
  right_ankle_roll_joint: 50,
  waist_yaw_joint: 88,
  waist_roll_joint: 50,
  waist_pitch_joint: 50,
  left_shoulder_pitch_joint: 25,
  left_shoulder_roll_joint: 25,
  left_shoulder_yaw_joint: 25,
  left_elbow_joint: 25,
  left_wrist_roll_joint: 25,
  left_wrist_pitch_joint: 5,
  left_wrist_yaw_joint: 5,
  right_shoulder_pitch_joint: 25,
  right_shoulder_roll_joint: 25,
  right_shoulder_yaw_joint: 25,
  right_elbow_joint: 25,
  right_wrist_roll_joint: 25,
  right_wrist_pitch_joint: 5,
  right_wrist_yaw_joint: 5,
};

function buildJointToActuatorMap(mujoco: MujocoModule, model: any): Map<number, number> {
  const map = new Map<number, number>();
  for (let i = 0; i < model.nu; i++) {
    if (model.actuator_trntype[i] === mujoco.mjtTrn.mjTRN_JOINT.value) {
      const jointId = model.actuator_trnid[i * 2];
      map.set(jointId, i);
    }
  }
  return map;
}

function getName(namesArray: Uint8Array, adr: number): string {
  const decoder = new TextDecoder('utf-8');
  let end = adr;
  while (end < namesArray.length && namesArray[end] !== 0) end += 1;
  return decoder.decode(namesArray.subarray(adr, end));
}

function getJointNames(model: any): string[] {
  let names = jointNameCache.get(model);
  if (!names) {
    const namesArray = new Uint8Array(model.names);
    names = Array.from({ length: model.njnt }, (_, i) => getName(namesArray, model.name_jntadr[i]));
    jointNameCache.set(model, names);
  }
  return names;
}

export function applyPDControl(
  mujoco: MujocoModule,
  model: any,
  data: any,
  qposRef: Float32Array,
  qvelRef: Float32Array,
  options: ControllerOptions,
): void {
  let jointToActuator = jointToActuatorCache.get(model);
  if (!jointToActuator) {
    jointToActuator = buildJointToActuatorMap(mujoco, model);
    jointToActuatorCache.set(model, jointToActuator);
  }

  // Zero out any previously applied controls / forces.
  for (let i = 0; i < model.nu; i++) {
    data.ctrl[i] = 0.0;
  }
  for (let i = 0; i < model.nv; i++) {
    data.qfrc_applied[i] = 0.0;
  }

  for (let j = 0; j < model.njnt; j++) {
    const dofAdr = model.jnt_dofadr[j];
    const qposAdr = model.jnt_qposadr[j];
    const type = model.jnt_type[j];
    if (type === mujoco.mjtJoint.mjJNT_FREE.value) continue; // floating base

    const jointName = getJointNames(model)[j] ?? '';
    const kp = (WBC_KP[jointName] ?? 80) * options.kp;
    const kd = (WBC_KD[jointName] ?? 2) * options.kd;
    const limit = (TORQUE_LIMIT[jointName] ?? options.maxTorque) * options.maxTorque;
    const errPos = Number(qposRef[qposAdr] ?? 0) - Number(data.qpos[qposAdr] ?? 0);
    const errVel = Number(qvelRef[dofAdr] ?? 0) - Number(data.qvel[dofAdr] ?? 0);
    const torque = kp * errPos + kd * errVel;

    const actIdx = jointToActuator.get(j);
    if (actIdx !== undefined) {
      data.ctrl[actIdx] = clampActuatorControl(model, actIdx, torque, limit);
    } else {
      // Fallback if no actuator is defined for this joint.
      data.qfrc_applied[dofAdr] = clamp(torque, -limit, limit);
    }
  }
}

function clampActuatorControl(
  model: any,
  actuatorIndex: number,
  value: number,
  fallbackLimit: number,
): number {
  // The MuJoCo WASM binding may throw for memory_view<bool> fields such as
  // actuator_ctrllimited, so use the numeric ctrlrange when it is meaningful.
  const range = model.actuator_ctrlrange;
  if (!range || range.length < (actuatorIndex + 1) * 2) {
    if (Number.isFinite(fallbackLimit) && fallbackLimit > 0) {
      return clamp(value, -fallbackLimit, fallbackLimit);
    }
    return value;
  }

  const min = range[actuatorIndex * 2];
  const max = range[(actuatorIndex * 2) + 1];
  if (!Number.isFinite(min) || !Number.isFinite(max) || min >= max) {
    if (Number.isFinite(fallbackLimit) && fallbackLimit > 0) {
      return clamp(value, -fallbackLimit, fallbackLimit);
    }
    return value;
  }
  return clamp(value, min, max);
}

function clamp(value: number, min: number, max: number) {
  return Math.min(Math.max(value, min), max);
}

export function setStateKinematic(
  model: any,
  data: any,
  qposRef: Float32Array,
  qvelRef?: Float32Array,
): void {
  for (let i = 0; i < model.nq && i < qposRef.length; i++) {
    data.qpos[i] = qposRef[i];
  }
  if (qvelRef) {
    for (let i = 0; i < model.nv && i < qvelRef.length; i++) {
      data.qvel[i] = qvelRef[i];
    }
  }
}
