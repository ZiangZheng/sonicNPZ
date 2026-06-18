import './style.css';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import loadMujoco from '@mujoco/mujoco';
import wasmUrl from '@mujoco/mujoco/mujoco.wasm?url';

import { loadMuJoCoScene, updateSceneTransforms, getBodyWorldTransform, type MuJoCoScene } from './mujocoScene';
import {
  loadMotionFromURL,
  loadMotionVariantsFromFile,
  sampleMotion,
  type MotionClip,
  type MotionLoadProgress,
  type MotionVariant,
} from './motion';
import { applyPDControl, setStateKinematic, DEFAULT_CONTROLLER_OPTIONS, type ControllerOptions } from './controller';
import { CameraWindow } from './cameras';
import { RealTimePlot } from './plots';
import { PolicyController } from './phpPolicyController.js';
import { HandFkWindow } from './handFkWindow';
import { SonicPolicyController } from './sonicPolicyController';

const MUJOCO_SCENE_PATH = '/working/scene.xml';
const DEFAULT_MOTION_URL = './motions/holdout_ep0071.npz';
const STAND_MOTION_URL = './motions/stand_sonic.json';
const SONIC_ENCODER_URL = './policies/sonic/model_encoder.onnx';
const SONIC_DECODER_URL = './policies/sonic/model_decoder.onnx';
const SONIC_BALANCE_URL = './policies/sonic/balance.onnx';
const BUILTIN_MOTIONS = [
  { label: 'HumanIDM holdout ep0071', url: './motions/holdout_ep0071.npz' },
  { label: 'Sonic official stand', url: STAND_MOTION_URL },
  { label: 'Sonic official squat', url: './motions/squat_001.json' },
];

type ModelId = 'sonic' | 'php';
type PlayMode = 'kinematic' | 'reference' | 'sonic';

interface DynamicsStats {
  steps: number;
  ctrlRms: number;
  ctrlMax: number;
  mode: PlayMode;
  rootAssist: boolean;
  reference: 'motion' | 'stand' | 'none';
}

interface ModelProfile {
  id: ModelId;
  label: string;
  sceneUrl: string;
  meshBaseUrl: string;
  defaultMotionUrl?: string;
  initialJointPos?: number[];
  policy?: {
    modelPath: string;
    depthModelPath: string | null;
    controlDt: number;
  };
}

const PHP_DEFAULT_JOINT_POS = [
  0.162997201, -0.0361181423, -0.0214254409, 0.267154634, -0.174296871, 0.212671682,
  0.282425106, -0.0584460497, -0.556104779, 0.126711249, -0.123827517, -0.190653816,
  0.000492588617, -0.0195334535, 0.428676069,
  -0.00628881808, 0.161155701, 0.236345276, 0.980316162, 0.15456377, 0.0774896815, 0.0205286704,
  -0.128641531, -0.0847690701, -0.255017966, 1.09530210, -0.134532213, 0.0875737667, 0.0601755157,
];

const MODEL_PROFILES: Record<ModelId, ModelProfile> = {
  sonic: {
    id: 'sonic',
    label: 'Sonic',
    sceneUrl: './assets/g1_wbc/g1_gear_wbc.xml',
    meshBaseUrl: './assets/g1_wbc/meshes/',
  },
  php: {
    id: 'php',
    label: 'PHP',
    sceneUrl: './assets/php/g1_with_terrain.xml',
    meshBaseUrl: './assets/g1/meshes/g1/',
    initialJointPos: PHP_DEFAULT_JOINT_POS,
    policy: {
      modelPath: './policies/php/student.onnx',
      depthModelPath: './policies/php/depth_backbone.onnx',
      controlDt: 0.02,
    },
  },
};

function getInitialModelId(): ModelId {
  const model = new URLSearchParams(window.location.search).get('model')?.toLowerCase();
  return model === 'php' ? 'php' : 'sonic';
}

async function setupVFS(mujoco: any, profile: ModelProfile) {
  mkdirp(mujoco, '/working');
  mkdirp(mujoco, '/working/meshes/g1');

  const parser = new DOMParser();
  const xmlSources: string[] = [];
  const loadedXml = new Set<string>();

  async function loadXml(url: string, vfsPath: string) {
    if (loadedXml.has(vfsPath)) return;
    loadedXml.add(vfsPath);

    const res = await fetch(url);
    if (!res.ok) {
      throw new Error(`Failed to fetch MuJoCo XML: ${url} (${res.status} ${res.statusText})`);
    }

    const text = normalizeMuJoCoXml(await res.text());
    mujoco.FS.writeFile(vfsPath, text);
    xmlSources.push(text);

    const xmlDoc = parser.parseFromString(text, 'text/xml');
    const includes = Array.from(xmlDoc.querySelectorAll('include[file]'))
      .map((el) => el.getAttribute('file'))
      .filter(Boolean) as string[];

    await Promise.all(includes.map((file) => {
      const childUrl = new URL(file, url).toString();
      const childPath = `${dirname(vfsPath)}/${basename(file)}`;
      return loadXml(childUrl, childPath);
    }));
  }

  await loadXml(new URL(profile.sceneUrl, window.location.href).toString(), MUJOCO_SCENE_PATH);

  const meshFiles = Array.from(new Set(xmlSources.flatMap((xml) => {
    const xmlDoc = parser.parseFromString(xml, 'text/xml');
    return Array.from(xmlDoc.querySelectorAll('mesh'))
      .map((el) => el.getAttribute('file'))
      .filter(Boolean) as string[];
  })));

  const meshBaseUrl = new URL(profile.meshBaseUrl, window.location.href);
  const missingMeshes: string[] = [];
  await Promise.all(meshFiles.map(async (file) => {
    const url = new URL(file, meshBaseUrl).toString();
    const res = await fetch(url);
    if (!res.ok) {
      missingMeshes.push(`${file} (${res.status})`);
      return;
    }
    const buffer = await res.arrayBuffer();
    const vfsPath = `/working/meshes/g1/${file}`;
    mkdirp(mujoco, dirname(vfsPath));
    mujoco.FS.writeFile(vfsPath, new Uint8Array(buffer));
  }));

  if (missingMeshes.length > 0) {
    throw new Error(`Missing MuJoCo mesh assets: ${missingMeshes.join(', ')}`);
  }
}

function normalizeMuJoCoXml(text: string): string {
  return text.replace(/\bmeshdir=(["']).*?\1/g, 'meshdir="meshes/g1"');
}

function mkdirp(mujoco: any, path: string) {
  const parts = path.split('/').filter(Boolean);
  let current = '';
  for (const part of parts) {
    current += `/${part}`;
    try {
      mujoco.FS.mkdir(current);
    } catch (err: any) {
      try {
        mujoco.FS.stat(current);
      } catch {
        throw err;
      }
    }
  }
}

function dirname(path: string): string {
  const idx = path.lastIndexOf('/');
  return idx <= 0 ? '/' : path.slice(0, idx);
}

function basename(path: string): string {
  return path.split('/').filter(Boolean).pop() ?? path;
}

function findFirstBodyId(mujoco: any, model: any, names: string[]): number {
  for (const name of names) {
    const id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY.value, name);
    if (id >= 0) return id;
  }
  return -1;
}

function createPhpSensorQuaternion(): THREE.Quaternion {
  const deg2rad = THREE.MathUtils.degToRad;
  const xAxis = new THREE.Vector3(1, 0, 0);
  const yAxis = new THREE.Vector3(0, 1, 0);
  const zAxis = new THREE.Vector3(0, 0, 1);
  const rpyDegToMjQuat = (rollDeg: number, pitchDeg: number, yawDeg: number) => {
    const qx = new THREE.Quaternion().setFromAxisAngle(xAxis, deg2rad(rollDeg));
    const qy = new THREE.Quaternion().setFromAxisAngle(yAxis, deg2rad(pitchDeg));
    const qz = new THREE.Quaternion().setFromAxisAngle(zAxis, deg2rad(yawDeg));
    return qz.multiply(qy).multiply(qx);
  };
  const mjQuatToThreeQuat = (qMj: THREE.Quaternion) =>
    new THREE.Quaternion(-qMj.x, -qMj.z, qMj.y, -qMj.w);

  const qOffsetMj = rpyDegToMjQuat(1, 27, 1);
  const qBaseMj = rpyDegToMjQuat(0, 0, -90);
  return mjQuatToThreeQuat(qOffsetMj.multiply(qBaseMj)).normalize();
}

async function init() {
  const app = document.getElementById('app')!;
  const activeModelId = getInitialModelId();
  const activeProfile = MODEL_PROFILES[activeModelId];

  // Canvas container
  const canvasContainer = document.createElement('div');
  canvasContainer.id = 'canvas-container';
  app.appendChild(canvasContainer);

  // Load MuJoCo WASM.
  const mujoco = await loadMujoco({ locateFile: (path: string) => (path === 'mujoco.wasm' ? wasmUrl : path) });
  await setupVFS(mujoco, activeProfile);

  const mjScene = await loadMuJoCoScene(mujoco, MUJOCO_SCENE_PATH);

  // Three.js renderer and scene.
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  canvasContainer.appendChild(renderer.domElement);

  mjScene.root.traverse((obj) => {
    if ((obj as THREE.Mesh).isMesh) {
      (obj as THREE.Mesh).frustumCulled = false;
    }
  });

  // Camera.
  const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.01, 200);
  camera.position.set(2.0, 1.7, 1.7);

  // Controls.
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0, 0.7, 0);
  controls.enableDamping = true;
  controls.dampingFactor = 0.05;

  // Lights.
  const ambient = new THREE.AmbientLight(0xffffff, 0.6);
  mjScene.root.add(ambient);

  const sun = new THREE.DirectionalLight(0xfff5e1, 2.5);
  sun.position.set(5, 8, 4);
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  sun.shadow.camera.near = 0.5;
  sun.shadow.camera.far = 50;
  sun.shadow.camera.left = -10;
  sun.shadow.camera.right = 10;
  sun.shadow.camera.top = 10;
  sun.shadow.camera.bottom = -10;
  mjScene.root.add(sun);

  const fill = new THREE.HemisphereLight(0x87ceeb, 0x5d4c3a, 0.4);
  mjScene.root.add(fill);

  // Sky gradient.
  const skyCanvas = document.createElement('canvas');
  skyCanvas.width = 512;
  skyCanvas.height = 512;
  const ctx = skyCanvas.getContext('2d')!;
  const grad = ctx.createLinearGradient(0, skyCanvas.height, 0, 0);
  grad.addColorStop(0.0, '#e8dcc8');
  grad.addColorStop(0.35, '#a7c4d9');
  grad.addColorStop(1.0, '#6d9cc8');
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, 512, 512);
  const skyTexture = new THREE.CanvasTexture(skyCanvas);
  skyTexture.colorSpace = THREE.SRGBColorSpace;

  // Add root to a separate scene so background applies only to main camera.
  const scene = new THREE.Scene();
  scene.background = skyTexture;
  scene.fog = new THREE.Fog(new THREE.Color(0xa7c4d9), 20, 90);
  scene.add(mjScene.root);

  // State.
  let motion: MotionClip | null = null;
  let standMotion: MotionClip | null = null;
  let uploadedVariants: MotionVariant[] = [];
  let activeMotionLabel = 'No reference';
  let isPlaying = false;
  let playMode: PlayMode = 'kinematic';
  let playSpeed = 1.0;
  let currentTime = 0;
  let simTime = 0;
  let lastFrameTime = performance.now();
  let controllerOptions: ControllerOptions = { ...DEFAULT_CONTROLLER_OPTIONS };
  let phpPolicy: any = null;
  let phpPolicyReady = false;
  let phpPolicyEnabled = activeModelId === 'php';
  let phpAutoForwardEnabled = activeModelId === 'php';
  let phpHighSpeedEnabled = true;
  let phpPolicyStepCounter = 0;
  let phpPolicyDecimation = 1;
  let latestReference: ReturnType<typeof sampleMotion> | null = null;
  const dynamicsStats: DynamicsStats = {
    steps: 0,
    ctrlRms: 0,
    ctrlMax: 0,
    mode: playMode,
    rootAssist: controllerOptions.rootAssist,
    reference: 'none',
  };
  const sonicPolicy = new SonicPolicyController(mujoco);
  let sonicPolicyReady = false;

  // Build UI.
  const ui = buildUI({
    activeModelId,
    initialPlaying: isPlaying,
    onPlayPause: () => {
      if (!motion && playMode !== 'sonic') {
        ui.motionStatusEl.textContent = 'No reference selected. Upload an NPZ/JSON or choose a built-in motion.';
        return isPlaying;
      }
      isPlaying = !isPlaying;
      return isPlaying;
    },
    onModelChange: (modelId) => {
      const params = new URLSearchParams(window.location.search);
      params.set('model', modelId);
      window.location.search = params.toString();
    },
    onReset: () => {
      currentTime = 0;
      simTime = 0;
      phpPolicyStepCounter = 0;
      sonicPolicy.reset(mjScene.model, mjScene.data);
      resetDynamicsStats(dynamicsStats, playMode, controllerOptions.rootAssist);
      if (motion && (playMode === 'kinematic' || playMode === 'reference')) {
        applyMotionReference(0);
      } else if (activeModelId === 'sonic' && playMode === 'sonic') {
        sonicPolicy.setInitialStandingState(mjScene.model, mjScene.data);
      } else {
        resetModelState(mjScene, mujoco, activeProfile);
        phpPolicy?.reset?.();
        if (phpPolicy) {
          phpPolicy.autoForward = phpAutoForwardEnabled;
          phpPolicy.highSpeedMode = phpHighSpeedEnabled;
          phpPolicy._updateCommandState?.();
        }
      }
    },
    onModeChange: (mode) => {
      if (mode === 'sonic') {
        simTime = currentTime;
        sonicPolicy.setInitialStandingState(mjScene.model, mjScene.data);
      } else if (mode === 'reference') {
        simTime = currentTime;
      } else {
        currentTime = simTime;
      }
      playMode = mode;
      resetDynamicsStats(dynamicsStats, playMode, controllerOptions.rootAssist);
      if (motion && (mode === 'kinematic' || mode === 'reference')) {
        applyMotionReference(currentTime);
      } else if (!motion && mode === 'kinematic') {
        resetModelState(mjScene, mujoco, activeProfile);
      }
    },
    onSpeedChange: (s) => (playSpeed = s),
    onClearMotion: () => {
      clearMotion();
    },
    onFile: async (file, onProgress) => {
      uploadedVariants = await loadMotionVariantsFromFile(file, onProgress);
      if (!uploadedVariants.length) throw new Error('No playable motion found in file.');
      setActiveMotion(uploadedVariants[0].motion, uploadedVariants[0].label);
      if (activeModelId === 'sonic') {
        playMode = 'reference';
        resetDynamicsStats(dynamicsStats, playMode, controllerOptions.rootAssist);
      }
      return uploadedVariants;
    },
    onUploadedVariant: (index) => {
      const variant = uploadedVariants[index];
      if (variant) setActiveMotion(variant.motion, variant.label);
    },
    onBuiltinMotion: async (url) => {
      const builtin = BUILTIN_MOTIONS.find((item) => item.url === url);
      const loaded = await loadMotionFromURL(url);
      setActiveMotion(loaded, builtin?.label ?? url);
      return loaded;
    },
    onSonicFiles: async (files) => {
      if (!ui.sonicStatusEl) return;
      ui.sonicStatusEl.textContent = 'Loading encoder/decoder...';
      try {
        const summary = await sonicPolicy.loadFromFiles(files);
        sonicPolicyReady = true;
        ui.sonicStatusEl.textContent = [
          'decoder policy ready',
          `enc in: ${summary.encoderInputs.join(', ') || '-'}`,
          `enc out: ${summary.encoderOutputs.join(', ') || '-'}`,
          `dec in: ${summary.decoderInputs.join(', ') || '-'}`,
          `dec out: ${summary.decoderOutputs.join(', ') || '-'}`,
        ].join(' · ');
      } catch (error) {
        console.error('Failed to load Sonic ONNX:', error);
        ui.sonicStatusEl.textContent = error instanceof Error ? error.message : String(error);
      }
    },
    onKpChange: (kp) => (controllerOptions.kp = kp),
    onKdChange: (kd) => (controllerOptions.kd = kd),
    onMaxTorqueChange: (maxTorque) => (controllerOptions.maxTorque = maxTorque),
    onRootAssistChange: (enabled) => {
      controllerOptions.rootAssist = enabled;
      dynamicsStats.rootAssist = enabled;
    },
    onPhpPolicyChange: (enabled) => (phpPolicyEnabled = enabled),
    onPhpAutoForwardChange: (enabled) => {
      phpAutoForwardEnabled = enabled;
      if (phpPolicy) {
        phpPolicy.autoForward = enabled;
        phpPolicy._updateCommandState?.();
      }
    },
    onPhpHighSpeedChange: (enabled) => {
      phpHighSpeedEnabled = enabled;
      if (phpPolicy) {
        phpPolicy.highSpeedMode = enabled;
        phpPolicy._updateCommandState?.();
      }
    },
    onPhpCommand: (key, pressed) => {
      phpPolicy?.setButtonPressed?.(key, pressed);
    },
    onSeek: (t) => {
      currentTime = t;
      simTime = t;
      if (motion) {
        applyMotionReference(currentTime);
      }
    },
  });
  app.appendChild(ui.root);

  // Camera windows.
  const rgbWindow = new CameraWindow(ui.rgbContainer, false);
  const depthWindow = new CameraWindow(ui.depthContainer, true);
  const handFkWindow = new HandFkWindow(ui.handContainer);
  handFkWindow.init().catch((err) => {
    console.error('Failed to initialize G1+FTP hand FK:', err);
  });
  const phpMainCameraOffset = new THREE.Vector3(-4.0, 1.5, 0.0);
  const phpSensorLocalPosition = new THREE.Vector3(0.01, 0.44, -0.01);
  const phpSensorLocalQuaternion = createPhpSensorQuaternion();
  const phpSensorAnchorBodyId = activeModelId === 'php'
    ? findFirstBodyId(mujoco, mjScene.model, ['torso_link', 'torso', 'trunk', 'waist_roll_link', 'pelvis'])
    : -1;

  if (activeProfile.policy) {
    ui.policyStatusEl!.textContent = 'Loading policy...';
    phpPolicy = new PolicyController(mujoco, activeProfile.policy);
    phpPolicy.init(mjScene.model).then(() => {
      phpPolicy.autoForward = phpAutoForwardEnabled;
      phpPolicy.highSpeedMode = phpHighSpeedEnabled;
      phpPolicy._updateCommandState?.();
      phpPolicyReady = true;
      phpPolicyDecimation = Math.max(1, Math.round(activeProfile.policy!.controlDt / mjScene.model.opt.timestep));
      ui.policyStatusEl!.textContent = 'Policy ready';
    }).catch((err: unknown) => {
      console.error('Failed to initialize PHP policy:', err);
      ui.policyStatusEl!.textContent = 'Policy failed';
    });
  }

  if (activeModelId === 'sonic') {
    ui.sonicStatusEl.textContent = 'Loading Sonic release policy...';
    sonicPolicy.loadFromURLs(SONIC_ENCODER_URL, SONIC_DECODER_URL, SONIC_BALANCE_URL).then((summary) => {
      sonicPolicyReady = true;
      sonicPolicy.reset(mjScene.model, mjScene.data);
      ui.sonicStatusEl.textContent = `WBC balance ready · decoder obs ${summary.decoderInputSize} · action ${summary.decoderOutputSize}`;
    }).catch((err: unknown) => {
      console.error('Failed to initialize Sonic policy:', err);
      ui.sonicStatusEl.textContent = err instanceof Error ? err.message : String(err);
    });
  }

  // Plots.
  const jointErrorPlot = ui.jointErrorCanvas ? new RealTimePlot(
    ui.jointErrorCanvas,
    'Joint Tracking Error (rad)',
    ['left_hip', 'right_hip', 'left_knee', 'right_knee'],
    ['#3870e8', '#00bcd4', '#f59e0b', '#ef4444'],
  ) : null;
  const rootHeightPlot = ui.rootHeightCanvas ? new RealTimePlot(
    ui.rootHeightCanvas,
    'Root Height (m)',
    ['ref', 'actual'],
    ['#00bcd4', '#3870e8'],
  ) : null;

  if (activeModelId === 'sonic') {
    loadMotionFromURL(STAND_MOTION_URL).then((stand) => {
      standMotion = stand;
    }).catch((err: unknown) => {
      console.error('Failed to load Sonic stand reference:', err);
    });
  }
  resetModelState(mjScene, mujoco, activeProfile);

  function sampleReference(time: number) {
    if (!motion) {
      dynamicsStats.reference = 'none';
      return;
    }
    const useStand = time > motion.duration;
    const ref = useStand && standMotion
      ? sampleMotion(standMotion, 0)
      : sampleMotion(motion, Math.min(time, motion.duration));
    dynamicsStats.reference = useStand ? 'stand' : 'motion';
    return ref;
  }

  function applyMotionReference(time: number) {
    const ref = sampleReference(time);
    if (!ref) return;
    latestReference = ref;
    setStateKinematic(mjScene.model, mjScene.data, ref.qpos, ref.qvel);
    clearControls(mjScene);
    mujoco.mj_forward(mjScene.model, mjScene.data);
    updateSceneTransforms(mjScene.model, mjScene.data, mjScene.bodies);
  }

  function stabilizeRootToReference(time: number) {
    const ref = sampleReference(time);
    if (!ref) return;
    const qposCount = Math.min(7, ref.qpos.length, mjScene.model.nq);
    const qvelCount = Math.min(6, ref.qvel.length, mjScene.model.nv);
    for (let i = 0; i < qposCount; i++) {
      mjScene.data.qpos[i] = ref.qpos[i];
    }
    for (let i = 0; i < qvelCount; i++) {
      mjScene.data.qvel[i] = ref.qvel[i];
    }
  }

  function updateMotionUI(m: MotionClip) {
    ui.motionStatusEl.textContent = `${activeMotionLabel} · ${m.times.length} frames · ${m.fps.toFixed(1)} fps`;
    ui.durationEl.textContent = `${m.duration.toFixed(2)}s`;
    ui.timeSlider.max = String(m.duration);
    ui.timeSlider.step = String(Math.max(m.duration / 1000, 0.001));
    ui.timeSlider.value = '0';
    ui.timeDisplay.textContent = '0.00s';
  }

  function setActiveMotion(nextMotion: MotionClip, label: string) {
    motion = nextMotion;
    activeMotionLabel = label;
    currentTime = 0;
    simTime = 0;
    latestReference = null;
    sonicPolicy.reset(mjScene.model, mjScene.data);
    resetDynamicsStats(dynamicsStats, playMode, controllerOptions.rootAssist);
    updateMotionUI(motion);
    if (playMode === 'kinematic' || playMode === 'reference') applyMotionReference(0);
  }

  function clearMotion() {
    motion = null;
    activeMotionLabel = 'No reference';
    currentTime = 0;
    simTime = 0;
    latestReference = null;
    resetDynamicsStats(dynamicsStats, playMode, controllerOptions.rootAssist);
    dynamicsStats.reference = 'none';
    ui.durationEl.textContent = '0.00s';
    ui.timeSlider.max = '1';
    ui.timeSlider.step = '0.01';
    ui.timeSlider.value = '0';
    ui.timeDisplay.textContent = '0.00s';
    ui.motionStatusEl.textContent = 'No reference selected.';
    resetModelState(mjScene, mujoco, activeProfile);
  }

  function updateMainCamera() {
    if (activeModelId !== 'php' || mjScene.pelvisBodyId < 0) return;
    const pelvisTransform = getBodyWorldTransform(mjScene.model, mjScene.data, mjScene.pelvisBodyId);
    camera.position.copy(pelvisTransform.position).add(phpMainCameraOffset);
    controls.target.copy(pelvisTransform.position);
  }

  function updateCameraWindows() {
    if (activeModelId === 'php' && phpSensorAnchorBodyId >= 0) {
      const anchorTransform = getBodyWorldTransform(mjScene.model, mjScene.data, phpSensorAnchorBodyId);
      const sensorPosition = phpSensorLocalPosition.clone()
        .applyQuaternion(anchorTransform.quaternion)
        .add(anchorTransform.position);
      const sensorQuaternion = anchorTransform.quaternion.clone().multiply(phpSensorLocalQuaternion);
      rgbWindow.updatePose(sensorPosition, sensorQuaternion);
      depthWindow.updatePose(sensorPosition, sensorQuaternion);
      return;
    }

    const headTransform = getBodyWorldTransform(mjScene.model, mjScene.data, mjScene.headBodyId);
    rgbWindow.updateCamera(headTransform.position, headTransform.quaternion);
    depthWindow.updateCamera(headTransform.position, headTransform.quaternion);
  }

  let plotAccumulator = 0;
  let statusAccumulator = 0;

  async function animate() {
    requestAnimationFrame(animate);

    const now = performance.now();
    const dt = Math.min((now - lastFrameTime) / 1000, 0.1);
    lastFrameTime = now;

    if (motion && isPlaying) {
      if (playMode === 'kinematic') {
        currentTime += dt * playSpeed;
        if (currentTime > motion.duration) currentTime = motion.duration;
        const ref = sampleMotion(motion, currentTime);
        latestReference = ref;
        setStateKinematic(mjScene.model, mjScene.data, ref.qpos, ref.qvel);
        mujoco.mj_forward(mjScene.model, mjScene.data);
        dynamicsStats.mode = playMode;
        dynamicsStats.rootAssist = controllerOptions.rootAssist;
        dynamicsStats.ctrlRms = 0;
        dynamicsStats.ctrlMax = 0;
      } else if (playMode === 'reference') {
        const step = mjScene.model.opt.timestep;
        const targetSimTime = simTime + dt * playSpeed;
        let steps = 0;
        while (simTime < targetSimTime && steps < 20) {
          const ref = sampleReference(simTime);
          if (!ref) break;
          latestReference = ref;
          applyPDControl(mujoco, mjScene.model, mjScene.data, ref.qpos, ref.qvel, controllerOptions);
          if (controllerOptions.rootAssist) stabilizeRootToReference(simTime);
          updateControlStats(dynamicsStats, mjScene.data, mjScene.model.nu);
          mujoco.mj_step(mjScene.model, mjScene.data);
          dynamicsStats.steps += 1;
          dynamicsStats.mode = playMode;
          dynamicsStats.rootAssist = controllerOptions.rootAssist;
          mujoco.mj_forward(mjScene.model, mjScene.data);
          simTime += step;
          steps++;
        }
        currentTime = Math.min(simTime, motion.duration);
      } else {
        // Sonic dynamics: keep MuJoCo live and let the policy continuously emit joint targets.
        const step = mjScene.model.opt.timestep;
        const targetSimTime = simTime + dt * playSpeed;
        let steps = 0;
        while (simTime < targetSimTime && steps < 20) {
          const ref = sampleReference(simTime);
          if (!ref) break;
          latestReference = ref;
          if (!sonicPolicyReady) break;
          await sonicPolicy.maybeUpdateAction(mjScene.model, mjScene.data).catch((err: unknown) => {
            console.error('Sonic policy inference error:', err);
          });
          sonicPolicy.applyControl(mjScene.model, mjScene.data);
          updateControlStats(dynamicsStats, mjScene.data, mjScene.model.nu);
          mujoco.mj_step(mjScene.model, mjScene.data);
          dynamicsStats.steps += 1;
          dynamicsStats.mode = playMode;
          dynamicsStats.rootAssist = controllerOptions.rootAssist;
          mujoco.mj_forward(mjScene.model, mjScene.data);
          simTime += step;
          steps++;
        }
        currentTime = Math.min(simTime, motion.duration);
      }

      ui.timeSlider.value = String(currentTime);
      ui.timeDisplay.textContent = `${currentTime.toFixed(2)}s`;
    } else if (!motion && isPlaying && activeModelId === 'sonic' && playMode === 'sonic') {
      const step = mjScene.model.opt.timestep;
      const targetSimTime = simTime + dt * playSpeed;
      let steps = 0;
      dynamicsStats.reference = 'stand';
      while (simTime < targetSimTime && steps < 20) {
        if (!sonicPolicyReady) break;
        await sonicPolicy.maybeUpdateAction(mjScene.model, mjScene.data).catch((err: unknown) => {
          console.error('Sonic policy inference error:', err);
        });
        sonicPolicy.applyControl(mjScene.model, mjScene.data);
        updateControlStats(dynamicsStats, mjScene.data, mjScene.model.nu);
        mujoco.mj_step(mjScene.model, mjScene.data);
        dynamicsStats.steps += 1;
        dynamicsStats.mode = playMode;
        dynamicsStats.rootAssist = controllerOptions.rootAssist;
        mujoco.mj_forward(mjScene.model, mjScene.data);
        simTime += step;
        steps++;
      }
    } else if (!motion && isPlaying && activeProfile.policy) {
      const step = mjScene.model.opt.timestep;
      const targetSimTime = simTime + dt * playSpeed;
      let steps = 0;
      while (simTime < targetSimTime && steps < 20) {
        if (activeProfile.policy) {
          if (!phpPolicyReady || !phpPolicyEnabled) break;
          if (phpPolicyStepCounter % phpPolicyDecimation === 0) {
            phpPolicy._updateCommandState?.();
            await phpPolicy.requestAction(mjScene.model, mjScene.data).catch((err: unknown) => {
              console.error('PHP policy inference error:', err);
            });
          }
          phpPolicy.applyControl(mjScene.model, mjScene.data);
          phpPolicyStepCounter++;
        }
        mujoco.mj_step(mjScene.model, mjScene.data);
        simTime += step;
        steps++;
      }
    }

    updateSceneTransforms(mjScene.model, mjScene.data, mjScene.bodies);
    handFkWindow.updateFromQpos(
      mjScene.data.qpos,
      currentTime || simTime,
      latestReference?.leftHandCmd,
      latestReference?.rightHandCmd,
    );
    updateMainCamera();
    updateCameraWindows();

    // Update plots every ~100ms.
    plotAccumulator += dt;
    statusAccumulator += dt;
    if (plotAccumulator > 0.1 && motion) {
      plotAccumulator = 0;
      const ref = latestReference ?? sampleMotion(motion, currentTime);
      const qpos = mjScene.data.qpos;
      const errors = [
        Math.abs(ref.qpos[7 + 0] - qpos[7 + 0]),
        Math.abs(ref.qpos[7 + 6] - qpos[7 + 6]),
        Math.abs(ref.qpos[7 + 3] - qpos[7 + 3]),
        Math.abs(ref.qpos[7 + 9] - qpos[7 + 9]),
      ];
      jointErrorPlot?.push(errors, currentTime.toFixed(1));
      rootHeightPlot?.push([ref.qpos[2], qpos[2]], currentTime.toFixed(1));
    }
    if (statusAccumulator > 0.2) {
      statusAccumulator = 0;
      updateStabilityReadout(ui.stabilityEl, mjScene.data, latestReference?.qpos);
      updateDynamicsReadout(ui.dynamicsEl, dynamicsStats);
    }

    controls.update();

    if (activeModelId === 'php' && phpPolicy) {
      const depthCapture = depthWindow.captureDepth(scene);
      if (depthCapture) {
        phpPolicy.setDepthImage(depthCapture.data, depthCapture.width, depthCapture.height);
      }
    }

    renderer.render(scene, camera);
    rgbWindow.render(scene);
    depthWindow.render(scene);
  }

  window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  });

  animate();
}

interface UIControls {
  activeModelId: ModelId;
  initialPlaying: boolean;
  onPlayPause: () => boolean;
  onReset: () => void;
  onModelChange: (modelId: ModelId) => void;
  onModeChange: (mode: PlayMode) => void;
  onSpeedChange: (speed: number) => void;
  onClearMotion: () => void;
  onFile: (file: File, onProgress: (progress: MotionLoadProgress) => void) => Promise<MotionVariant[]>;
  onUploadedVariant: (index: number) => void;
  onBuiltinMotion: (url: string) => Promise<MotionClip>;
  onSonicFiles: (files: FileList) => void;
  onKpChange: (kp: number) => void;
  onKdChange: (kd: number) => void;
  onMaxTorqueChange: (maxTorque: number) => void;
  onRootAssistChange: (enabled: boolean) => void;
  onPhpPolicyChange: (enabled: boolean) => void;
  onPhpAutoForwardChange: (enabled: boolean) => void;
  onPhpHighSpeedChange: (enabled: boolean) => void;
  onPhpCommand: (key: string, pressed: boolean) => void;
  onSeek: (time: number) => void;
}

function buildUI(c: UIControls) {
  const isSonic = c.activeModelId === 'sonic';
  const root = document.createElement('div');
  root.className = 'pointer-events-none absolute inset-0 z-10 p-4';

  // Header.
  const header = document.createElement('div');
  header.className = 'pointer-events-auto glass-panel px-5 py-3 inline-flex items-center gap-3 absolute left-4 top-4';
  header.innerHTML = `
    <div class="text-lg font-bold tracking-tight"><span class="text-cyan-400">${MODEL_PROFILES[c.activeModelId].label}</span> Visualizer</div>
    <div class="text-xs text-slate-400 hidden sm:block">Unitree G1 · MuJoCo WASM</div>
  `;
  root.appendChild(header);

  const leftDock = document.createElement('div');
  leftDock.className = 'pointer-events-auto absolute left-4 top-24 bottom-4 flex w-[330px] flex-col gap-4 overflow-y-auto pr-1';

  const rightDock = document.createElement('div');
  rightDock.className = 'pointer-events-auto absolute right-4 top-4 bottom-4 flex w-[330px] flex-col gap-4 overflow-y-auto pl-1';

  // Playback panel.
  const playbackPanel = document.createElement('div');
  playbackPanel.className = 'glass-panel p-4 flex flex-col gap-3 min-w-[280px] max-w-[360px]';

  const row1 = document.createElement('div');
  row1.className = 'flex items-center gap-2';
  const playBtn = document.createElement('button');
  playBtn.className = c.initialPlaying ? 'glass-button active' : 'glass-button';
  playBtn.textContent = c.initialPlaying ? 'Pause' : 'Play';
  playBtn.onclick = () => {
    const playing = c.onPlayPause();
    playBtn.textContent = playing ? 'Pause' : 'Play';
    playBtn.classList.toggle('active', playing);
  };
  const resetBtn = document.createElement('button');
  resetBtn.className = 'glass-button';
  resetBtn.textContent = 'Reset';
  resetBtn.onclick = c.onReset;
  row1.append(playBtn, resetBtn);

  const modelRow = document.createElement('div');
  modelRow.className = 'flex items-center gap-2 text-sm';
  modelRow.innerHTML = '<span class="text-slate-300 w-14">Model</span>';
  const modelSelect = document.createElement('select');
  modelSelect.className = 'glass-input flex-1';
  modelSelect.innerHTML = `
    <option value="sonic">Sonic</option>
    <option value="php">PHP</option>
  `;
  modelSelect.value = c.activeModelId;
  modelSelect.onchange = () => c.onModelChange(modelSelect.value as ModelId);
  modelRow.append(modelSelect);

  const modeRow = document.createElement('div');
  modeRow.className = 'flex gap-2';
  const kinBtn = document.createElement('button');
  kinBtn.className = 'glass-button active flex-1 px-3 text-xs whitespace-nowrap';
  kinBtn.textContent = 'Kin';
  const refDynBtn = document.createElement('button');
  refDynBtn.className = 'glass-button flex-1 px-3 text-xs whitespace-nowrap';
  refDynBtn.textContent = 'Ref Dyn';
  const policyBtn = document.createElement('button');
  policyBtn.className = 'glass-button flex-1 px-3 text-xs whitespace-nowrap';
  policyBtn.textContent = 'Stand';
  const setModeButtons = (mode: PlayMode) => {
    kinBtn.classList.toggle('active', mode === 'kinematic');
    refDynBtn.classList.toggle('active', mode === 'reference');
    policyBtn.classList.toggle('active', mode === 'sonic');
  };
  kinBtn.onclick = () => {
    c.onModeChange('kinematic');
    setModeButtons('kinematic');
  };
  refDynBtn.onclick = () => {
    c.onModeChange('reference');
    setModeButtons('reference');
  };
  policyBtn.onclick = () => {
    c.onModeChange('sonic');
    setModeButtons('sonic');
  };
  modeRow.append(kinBtn, refDynBtn, policyBtn);

  const speedRow = document.createElement('div');
  speedRow.className = 'flex items-center gap-2 text-sm';
  speedRow.innerHTML = '<span class="text-slate-300 w-14">Speed</span>';
  const speedInput = document.createElement('input');
  speedInput.type = 'range';
  speedInput.min = '0.1';
  speedInput.max = '2.0';
  speedInput.step = '0.1';
  speedInput.value = '1.0';
  speedInput.className = 'glass-range flex-1';
  const speedVal = document.createElement('span');
  speedVal.className = 'text-slate-300 w-8 text-right';
  speedVal.textContent = '1.0x';
  speedInput.oninput = () => {
    const v = parseFloat(speedInput.value);
    speedVal.textContent = v.toFixed(1) + 'x';
    c.onSpeedChange(v);
  };
  speedRow.append(speedInput, speedVal);

  const timeRow = document.createElement('div');
  timeRow.className = 'flex items-center gap-2 text-sm';
  const timeDisplay = document.createElement('span');
  timeDisplay.className = 'text-cyan-300 font-mono w-14';
  timeDisplay.textContent = '0.00s';
  const timeSlider = document.createElement('input');
  timeSlider.type = 'range';
  timeSlider.min = '0';
  timeSlider.max = '1';
  timeSlider.step = '0.01';
  timeSlider.value = '0';
  timeSlider.className = 'glass-range flex-1';
  timeSlider.oninput = () => {
    c.onSeek(parseFloat(timeSlider.value));
  };
  const durationEl = document.createElement('span');
  durationEl.className = 'text-slate-400 w-14 text-right';
  durationEl.textContent = '0.00s';
  timeRow.append(timeDisplay, timeSlider, durationEl);

  const fileRow = document.createElement('div');
  fileRow.className = 'flex flex-col gap-2';
  const fileInput = document.createElement('input');
  fileInput.type = 'file';
  fileInput.accept = '.json,.npz';
  fileInput.className = 'hidden';
  const uploadBtn = document.createElement('button');
  uploadBtn.className = 'glass-button w-full';
  uploadBtn.textContent = 'Upload reference motion';
  const uploadProgress = document.createElement('div');
  uploadProgress.className = 'upload-progress hidden';
  const uploadProgressFill = document.createElement('div');
  uploadProgress.append(uploadProgressFill);
  const motionStatusEl = document.createElement('div');
  motionStatusEl.className = 'text-[11px] leading-snug text-slate-400 max-w-[320px] break-words';
  motionStatusEl.textContent = 'No reference selected.';
  fileInput.onchange = async () => {
    const file = fileInput.files?.[0];
    if (!file) return;
    uploadBtn.disabled = true;
    uploadBtn.textContent = 'Loading...';
    uploadProgress.classList.remove('hidden');
    uploadProgressFill.style.width = '0%';
    motionStatusEl.textContent = `Reading ${file.name}...`;
    try {
      const variants = await c.onFile(file, (progress) => {
        uploadProgressFill.style.width = `${Math.round(progress.ratio * 100)}%`;
        const loadedMb = progress.loaded / (1024 * 1024);
        const totalMb = progress.total / (1024 * 1024);
        motionStatusEl.textContent = `${progress.phase} ${file.name} · ${loadedMb.toFixed(1)} / ${totalMb.toFixed(1)} MB`;
      });
      setModeButtons('reference');
      uploadedOption.hidden = false;
      uploadedOption.textContent = `Uploaded: ${file.name}`;
      motionSelect.value = '__uploaded__';
      trackSelect.innerHTML = variants.map((variant, index) =>
        `<option value="${index}">${variant.label}</option>`,
      ).join('');
      trackRow.classList.toggle('hidden', variants.length <= 1);
      trackSelect.value = '0';
      const first = variants[0].motion;
      motionStatusEl.textContent = `${variants[0].label} · ${first.times.length} frames · ${first.duration.toFixed(2)}s · ready`;
    } catch (error) {
      console.error('Failed to load reference motion:', error);
      motionStatusEl.textContent = error instanceof Error ? error.message : String(error);
      uploadProgressFill.style.width = '0%';
    } finally {
      uploadBtn.disabled = false;
      uploadBtn.textContent = 'Upload reference motion';
      fileInput.value = '';
    }
  };
  uploadBtn.onclick = () => fileInput.click();
  fileRow.append(fileInput, uploadBtn, uploadProgress, motionStatusEl);

  const builtinRow = document.createElement('div');
  builtinRow.className = 'flex items-center gap-2 text-sm';
  builtinRow.innerHTML = '<span class="text-slate-300 w-14">Motion</span>';
  const builtinSelect = document.createElement('select');
  const motionSelect = builtinSelect;
  motionSelect.className = 'glass-input flex-1';
  motionSelect.innerHTML = [
    '<option value="__none__">No reference</option>',
    ...BUILTIN_MOTIONS.map((motionItem) => `<option value="${motionItem.url}">${motionItem.label}</option>`),
    '<option value="__uploaded__" hidden>Uploaded file</option>',
  ].join('');
  const uploadedOption = motionSelect.querySelector('option[value="__uploaded__"]') as HTMLOptionElement;
  motionSelect.value = '__none__';
  motionSelect.onchange = async () => {
    const value = motionSelect.value;
    trackRow.classList.add('hidden');
    if (value === '__none__') {
      c.onClearMotion();
      return;
    }
    if (value === '__uploaded__') {
      trackRow.classList.toggle('hidden', trackSelect.options.length <= 1);
      c.onUploadedVariant(Number(trackSelect.value || 0));
      return;
    }
    motionStatusEl.textContent = 'Loading built-in motion...';
    try {
      const loaded = await c.onBuiltinMotion(value);
      motionStatusEl.textContent = `${motionSelect.selectedOptions[0]?.textContent ?? value} · ${loaded.times.length} frames · ${loaded.duration.toFixed(2)}s · ready`;
    } catch (error) {
      console.error('Failed to load built-in motion:', error);
      motionStatusEl.textContent = error instanceof Error ? error.message : String(error);
    }
  };
  builtinRow.append(motionSelect);

  const trackRow = document.createElement('div');
  trackRow.className = 'hidden flex items-center gap-2 text-sm';
  trackRow.innerHTML = '<span class="text-slate-300 w-14">Track</span>';
  const trackSelect = document.createElement('select');
  trackSelect.className = 'glass-input flex-1';
  trackSelect.onchange = () => {
    c.onUploadedVariant(Number(trackSelect.value));
    motionStatusEl.textContent = `Selected ${trackSelect.selectedOptions[0]?.textContent ?? 'uploaded track'}`;
  };
  trackRow.append(trackSelect);

  const sonicOnnxRow = document.createElement('div');
  sonicOnnxRow.className = 'flex flex-col gap-2';
  const sonicInput = document.createElement('input');
  sonicInput.type = 'file';
  sonicInput.accept = '.onnx';
  sonicInput.multiple = true;
  sonicInput.className = 'hidden';
  sonicInput.onchange = () => {
    if (sonicInput.files?.length) c.onSonicFiles(sonicInput.files);
  };
  const sonicUploadBtn = document.createElement('button');
  sonicUploadBtn.className = 'glass-button w-full';
  sonicUploadBtn.textContent = 'Override Sonic ONNX';
  sonicUploadBtn.onclick = () => sonicInput.click();
  const sonicStatusEl = document.createElement('div');
  sonicStatusEl.className = 'text-[11px] leading-snug text-slate-400 max-w-[320px] break-words';
  sonicStatusEl.textContent = 'Bundled Sonic ONNX loading automatically...';
  sonicOnnxRow.append(sonicInput, sonicUploadBtn, sonicStatusEl);

  playbackPanel.append(row1, modelRow);
  if (isSonic) playbackPanel.append(modeRow);
  playbackPanel.append(speedRow);
  if (isSonic) playbackPanel.append(builtinRow, trackRow, timeRow, fileRow, sonicOnnxRow);

  // Options panel.
  const optionsPanel = document.createElement('div');
  optionsPanel.className = 'glass-panel p-4 flex flex-col gap-3 min-w-[220px]';
  optionsPanel.innerHTML = `
    <div class="text-sm font-semibold text-slate-200">Sonic / WBC Dynamics</div>
  `;
  const kpRow = createNumberRow('Gain', DEFAULT_CONTROLLER_OPTIONS.kp, 0, 2, 0.05, c.onKpChange);
  const kdRow = createNumberRow('Damp', DEFAULT_CONTROLLER_OPTIONS.kd, 0, 2, 0.05, c.onKdChange);
  const maxTorqueRow = createNumberRow('Torque', DEFAULT_CONTROLLER_OPTIONS.maxTorque, 0, 1.5, 0.05, c.onMaxTorqueChange);
  const rootAssistRow = createCheckboxRow('Root assist', DEFAULT_CONTROLLER_OPTIONS.rootAssist, c.onRootAssistChange);
  const stabilityEl = document.createElement('div');
  stabilityEl.className = 'text-xs leading-snug text-slate-400';
  stabilityEl.textContent = 'height -- · tilt -- · mode free root';
  const dynamicsEl = document.createElement('div');
  dynamicsEl.className = 'text-xs leading-snug text-slate-400';
  dynamicsEl.textContent = 'kinematic qpos · ref none · steps 0 · ctrl 0.0';
  optionsPanel.append(kpRow, kdRow, maxTorqueRow, rootAssistRow, stabilityEl, dynamicsEl);

  const phpOptionsPanel = document.createElement('div');
  phpOptionsPanel.className = 'glass-panel p-4 flex flex-col gap-3 min-w-[220px]';
  phpOptionsPanel.innerHTML = `
    <div class="text-sm font-semibold text-slate-200">PHP Policy</div>
    <div class="text-xs text-slate-400" id="policy-status">Policy disabled</div>
  `;
  const policyRow = createCheckboxRow('Enabled', true, c.onPhpPolicyChange);
  const autoForwardRow = createCheckboxRow('Auto forward', !isSonic, c.onPhpAutoForwardChange);
  const highSpeedRow = createCheckboxRow('High speed', true, c.onPhpHighSpeedChange);
  const commandPad = createPhpCommandPad(c.onPhpCommand);
  phpOptionsPanel.append(policyRow, autoForwardRow, highSpeedRow, commandPad);

  // Camera windows panel.
  const cameraPanel = document.createElement('div');
  cameraPanel.className = 'glass-panel p-3 flex flex-col gap-2';
  cameraPanel.innerHTML = `
    <div class="text-xs font-semibold text-slate-300">RGB Camera</div>
    <div class="camera-window" id="rgb-container"></div>
    <div class="text-xs font-semibold text-slate-300 mt-1">Depth Camera</div>
    <div class="camera-window" id="depth-container"></div>
    <div class="text-xs font-semibold text-slate-300 mt-1">G1+FTP Hand FK</div>
    <div class="camera-window" id="hand-fk-container"></div>
  `;

  // Plots panel.
  const plotsPanel = document.createElement('div');
  plotsPanel.className = 'glass-panel p-3 flex flex-col gap-2';
  plotsPanel.innerHTML = `
    <div class="plot-container"><canvas id="joint-error-canvas"></canvas></div>
    <div class="plot-container"><canvas id="root-height-canvas"></canvas></div>
  `;

  leftDock.append(playbackPanel);
  if (isSonic) leftDock.append(optionsPanel);
  if (!isSonic) leftDock.append(phpOptionsPanel);
  rightDock.append(cameraPanel);
  if (isSonic) rightDock.append(plotsPanel);
  root.append(leftDock, rightDock);

  return {
    root,
    rgbContainer: cameraPanel.querySelector('#rgb-container') as HTMLDivElement,
    depthContainer: cameraPanel.querySelector('#depth-container') as HTMLDivElement,
    handContainer: cameraPanel.querySelector('#hand-fk-container') as HTMLDivElement,
    jointErrorCanvas: plotsPanel.querySelector('#joint-error-canvas') as HTMLCanvasElement | null,
    rootHeightCanvas: plotsPanel.querySelector('#root-height-canvas') as HTMLCanvasElement | null,
    policyStatusEl: phpOptionsPanel.querySelector('#policy-status') as HTMLDivElement | null,
    sonicStatusEl,
    motionStatusEl,
    stabilityEl,
    dynamicsEl,
    timeDisplay,
    timeSlider,
    durationEl,
  };
}

function updateControlStats(stats: DynamicsStats, data: any, nu: number) {
  let sumSq = 0;
  let maxAbs = 0;
  for (let i = 0; i < nu; i += 1) {
    const value = Number(data.ctrl[i] ?? 0);
    sumSq += value * value;
    maxAbs = Math.max(maxAbs, Math.abs(value));
  }
  stats.ctrlRms = nu > 0 ? Math.sqrt(sumSq / nu) : 0;
  stats.ctrlMax = maxAbs;
}

function resetDynamicsStats(stats: DynamicsStats, mode: PlayMode, rootAssist: boolean) {
  stats.steps = 0;
  stats.ctrlRms = 0;
  stats.ctrlMax = 0;
  stats.mode = mode;
  stats.rootAssist = rootAssist;
  stats.reference = 'none';
}

function updateDynamicsReadout(element: HTMLElement, stats: DynamicsStats) {
  const mode = stats.mode === 'sonic'
    ? 'policy stand mj_step'
    : stats.mode === 'reference'
      ? stats.rootAssist ? 'ref PD mj_step + root assist' : 'ref PD free-root mj_step'
      : 'kinematic qpos';
  element.textContent = `${mode} · ref ${stats.reference} · steps ${stats.steps} · ctrl rms ${stats.ctrlRms.toFixed(1)} max ${stats.ctrlMax.toFixed(1)}`;
}

function updateStabilityReadout(element: HTMLElement, data: any, qposRef?: Float32Array) {
  const q = data.qpos;
  const height = Number(q[2] ?? 0);
  const refHeight = Number(qposRef?.[2] ?? 0.78);
  const quat = [Number(q[3] ?? 1), Number(q[4] ?? 0), Number(q[5] ?? 0), Number(q[6] ?? 0)];
  const upZ = quatUpZ(quat);
  const tilt = Math.acos(Math.max(-1, Math.min(1, upZ))) * 180 / Math.PI;
  const stableHeight = Math.max(0.25, refHeight - 0.18);
  const recoveringHeight = Math.max(0.18, refHeight - 0.35);
  const tone = height > stableHeight && tilt < 35
    ? 'stable'
    : height > recoveringHeight && tilt < 65
      ? 'recovering'
      : 'fallen';
  element.textContent = `height ${height.toFixed(2)} m · tilt ${tilt.toFixed(0)} deg · ${tone}`;
  element.dataset.tone = tone;
}

function quatUpZ(q: number[]) {
  const [w, x, y, z] = q;
  return 1 - 2 * (x * x + y * y);
}

function resetModelState(scene: MuJoCoScene, mujoco: any, profile: ModelProfile) {
  mujoco.mj_resetData(scene.model, scene.data);
  if (profile.initialJointPos) {
    for (let i = 0; i < profile.initialJointPos.length; i++) {
      scene.data.qpos[7 + i] = profile.initialJointPos[i];
      scene.data.qvel[6 + i] = 0;
    }
  }
  clearControls(scene);
  mujoco.mj_forward(scene.model, scene.data);
  updateSceneTransforms(scene.model, scene.data, scene.bodies);
}

function clearControls(scene: MuJoCoScene) {
  for (let i = 0; i < scene.model.nu; i++) {
    scene.data.ctrl[i] = 0;
  }
  for (let i = 0; i < scene.model.nv; i++) {
    scene.data.qfrc_applied[i] = 0;
  }
}

function createNumberRow(
  label: string,
  initial: number,
  min: number,
  max: number,
  step: number,
  onChange: (v: number) => void,
) {
  const row = document.createElement('div');
  row.className = 'flex items-center gap-2 text-sm';
  row.innerHTML = `<span class="text-slate-300 w-10">${label}</span>
    <input type="number" class="glass-input w-16 text-right" min="${min}" max="${max}" step="${step}" value="${initial}">`;
  const input = row.querySelector('input')!;
  input.onchange = () => onChange(parseFloat(input.value));
  return row;
}

function createPhpCommandPad(onCommand: (key: string, pressed: boolean) => void) {
  const pad = document.createElement('div');
  pad.className = 'grid grid-cols-3 gap-2 pt-1';

  const buttonSpecs = [
    { key: 'q', label: 'Q' },
    { key: 'w', label: 'W' },
    { key: 'e', label: 'E' },
    { key: 'a', label: 'A' },
    { key: '', label: '' },
    { key: 'd', label: 'D' },
  ];

  for (const spec of buttonSpecs) {
    if (!spec.key) {
      const spacer = document.createElement('div');
      pad.append(spacer);
      continue;
    }

    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'glass-button px-0 py-2 select-none';
    button.textContent = spec.label;
    button.style.touchAction = 'none';
    const setPressed = (pressed: boolean) => {
      button.classList.toggle('active', pressed);
      onCommand(spec.key, pressed);
    };
    button.onpointerdown = (event) => {
      button.setPointerCapture(event.pointerId);
      setPressed(true);
      event.preventDefault();
    };
    button.onpointerup = (event) => {
      setPressed(false);
      button.releasePointerCapture(event.pointerId);
      event.preventDefault();
    };
    button.onpointercancel = () => setPressed(false);
    button.onpointerleave = () => setPressed(false);
    pad.append(button);
  }

  return pad;
}

function createCheckboxRow(
  label: string,
  initial: boolean,
  onChange: (v: boolean) => void,
) {
  const row = document.createElement('label');
  row.className = 'flex items-center justify-between gap-3 text-sm text-slate-300';
  const span = document.createElement('span');
  span.textContent = label;
  const input = document.createElement('input');
  input.type = 'checkbox';
  input.checked = initial;
  input.className = 'h-4 w-4 accent-blue-500';
  input.onchange = () => onChange(input.checked);
  row.append(span, input);
  return row;
}

init().catch((err) => {
  console.error('Initialization failed:', err);
  document.body.innerHTML = `
    <div class="p-8 text-red-400">
      <h1 class="text-xl font-bold">Failed to load visualizer</h1>
      <pre class="mt-2 text-sm">${err?.message || err}</pre>
    </div>`;
});
