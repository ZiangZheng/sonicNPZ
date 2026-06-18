import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import URDFLoader from 'urdf-loader';

const G1_FTP_URDF = '/robots/g1_ftp/g1_inspire_FTP.urdf';

const G1_JOINTS_29DOF = [
  'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint',
  'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
  'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint',
  'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint',
  'waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint',
  'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint',
  'left_elbow_joint', 'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint',
  'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint',
  'right_elbow_joint', 'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint',
];

const FTP_FINGER_JOINT_SUFFIXES = [
  'thumb_1', 'thumb_2', 'thumb_3', 'thumb_4',
  'index_1', 'index_2',
  'middle_1', 'middle_2',
  'ring_1', 'ring_2',
  'little_1', 'little_2',
];

type UrdfRobot = THREE.Object3D & {
  joints?: Record<string, { setJointValue: (value: number) => void }>;
};

export class HandFkWindow {
  private readonly container: HTMLElement;
  private readonly scene = new THREE.Scene();
  private readonly renderer: THREE.WebGLRenderer;
  private readonly camera: THREE.PerspectiveCamera;
  private readonly controls: OrbitControls;
  private readonly resizeObserver: ResizeObserver;
  private robot?: UrdfRobot;
  private frame = 0;

  constructor(container: HTMLElement) {
    this.container = container;
    const canvas = document.createElement('canvas');
    canvas.className = 'w-full h-full block';
    container.appendChild(canvas);

    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
    this.renderer.setSize(container.clientWidth, container.clientHeight);
    this.scene.background = new THREE.Color(0x05090b);

    this.camera = new THREE.PerspectiveCamera(42, 1, 0.02, 30);
    this.camera.up.set(0, 0, 1);
    this.camera.position.set(0.85, -1.05, 1.05);

    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.target.set(0.1, 0, 0.78);
    this.controls.enableDamping = true;

    this.addLights();
    this.addFloor();

    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(container);
    this.resize();
    this.animate();
  }

  async init() {
    const loader = new URDFLoader();
    this.robot = await new Promise<UrdfRobot>((resolve, reject) => {
      loader.load(G1_FTP_URDF, resolve, undefined, reject);
    });
    this.robot.traverse((object) => {
      if (object instanceof THREE.Mesh) {
        object.castShadow = true;
        object.receiveShadow = true;
        object.frustumCulled = false;
        object.visible = this.hasHandAncestor(object);
      }
    });
    this.scene.add(this.robot);
  }

  updateFromQpos(
    qpos: ArrayLike<number>,
    time: number,
    leftHandCmd?: ArrayLike<number>,
    rightHandCmd?: ArrayLike<number>,
  ) {
    if (!this.robot) return;
    this.frame += 1;
    this.robot.position.set(Number(qpos[0] ?? 0), Number(qpos[1] ?? 0), Number(qpos[2] ?? 0));
    this.robot.quaternion.set(
      Number(qpos[4] ?? 0),
      Number(qpos[5] ?? 0),
      Number(qpos[6] ?? 0),
      Number(qpos[3] ?? 1),
    ).normalize();
    for (let i = 0; i < G1_JOINTS_29DOF.length; i += 1) {
      this.robot.joints?.[G1_JOINTS_29DOF[i]]?.setJointValue(Number(qpos[7 + i] ?? 0));
    }
    if (leftHandCmd || rightHandCmd) {
      if (leftHandCmd) this.applyFingerCommand('left', leftHandCmd);
      if (rightHandCmd) this.applyFingerCommand('right', rightHandCmd);
    } else {
      this.applySyntheticFingers(time);
    }
  }

  private applyFingerCommand(side: 'left' | 'right', values: ArrayLike<number>) {
    for (let i = 0; i < FTP_FINGER_JOINT_SUFFIXES.length; i += 1) {
      this.robot?.joints?.[`${side}_${FTP_FINGER_JOINT_SUFFIXES[i]}_joint`]?.setJointValue(Number(values[i] ?? 0));
    }
  }

  private applySyntheticFingers(time: number) {
    const left = 0.35 + 0.35 * Math.sin(time * 2.7);
    const right = 0.35 + 0.35 * Math.cos(time * 2.5);
    for (const side of ['left', 'right'] as const) {
      const base = side === 'left' ? left : right;
      for (let i = 0; i < FTP_FINGER_JOINT_SUFFIXES.length; i += 1) {
        const value = Math.max(0, base + 0.08 * Math.sin(time * 4 + i * 0.35));
        this.robot?.joints?.[`${side}_${FTP_FINGER_JOINT_SUFFIXES[i]}_joint`]?.setJointValue(value);
      }
    }
  }

  private hasHandAncestor(object: THREE.Object3D) {
    let cursor: THREE.Object3D | null = object;
    while (cursor) {
      if (/(hand|wrist|palm|thumb|index|middle|ring|little|finger)/i.test(cursor.name)) return true;
      cursor = cursor.parent;
    }
    return false;
  }

  private addLights() {
    this.scene.add(new THREE.AmbientLight(0xcde9ff, 0.5));
    const key = new THREE.DirectionalLight(0xffffff, 1.6);
    key.position.set(-2.0, -3.0, 4.0);
    key.castShadow = true;
    this.scene.add(key);
    const fill = new THREE.DirectionalLight(0xffd7aa, 0.55);
    fill.position.set(3.0, 2.0, 2.0);
    this.scene.add(fill);
  }

  private addFloor() {
    const floor = new THREE.Mesh(
      new THREE.PlaneGeometry(4, 4),
      new THREE.MeshStandardMaterial({ color: 0x1a2225, roughness: 0.9 }),
    );
    floor.receiveShadow = true;
    this.scene.add(floor);
  }

  private resize() {
    const width = Math.max(1, this.container.clientWidth);
    const height = Math.max(1, this.container.clientHeight);
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
  }

  private animate() {
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
    requestAnimationFrame(() => this.animate());
  }
}
