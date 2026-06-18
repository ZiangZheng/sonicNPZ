import * as THREE from 'three';

export class CameraWindow {
  canvas: HTMLCanvasElement;
  renderer: THREE.WebGLRenderer;
  camera: THREE.PerspectiveCamera;
  isDepth: boolean;
  depthMaterial: THREE.ShaderMaterial;
  depthInferenceMaterial: THREE.ShaderMaterial | null = null;
  depthInferenceScene: THREE.Scene | null = null;
  depthOrthoCamera: THREE.OrthographicCamera | null = null;
  depthInferenceTarget: THREE.WebGLRenderTarget | null = null;
  depthTarget: THREE.WebGLRenderTarget | null = null;
  depthPixels: Float32Array | null = null;
  depthFrame: Float32Array | null = null;
  depthCaptureSize = { width: 106, height: 60 };
  private depthCaptureWarned = false;
  private savedBackground: THREE.Color | null = null;

  constructor(container: HTMLElement, isDepth: boolean) {
    this.isDepth = isDepth;
    this.canvas = document.createElement('canvas');
    this.canvas.className = 'w-full h-full block';
    container.appendChild(this.canvas);

    this.renderer = new THREE.WebGLRenderer({
      canvas: this.canvas,
      antialias: false,
      alpha: false,
    });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
    this.renderer.setSize(container.clientWidth, container.clientHeight);

    this.camera = new THREE.PerspectiveCamera(
      58,
      isDepth ? this.depthCaptureSize.width / this.depthCaptureSize.height : container.clientWidth / container.clientHeight,
      isDepth ? 0.3 : 0.05,
      isDepth ? 3.0 : 20,
    );

    this.depthMaterial = new THREE.ShaderMaterial({
      uniforms: {
        near: { value: this.camera.near },
        far: { value: this.camera.far },
      },
      vertexShader: `
        varying float vDepth;
        void main() {
          vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
          vDepth = -mvPosition.z;
          gl_Position = projectionMatrix * mvPosition;
        }
      `,
      fragmentShader: `
        varying float vDepth;
        uniform float near;
        uniform float far;
        void main() {
          float gray = clamp((vDepth - near) / (far - near), 0.0, 1.0);
          gl_FragColor = vec4(vec3(gray), 1.0);
        }
      `,
      side: THREE.DoubleSide,
    });

    if (isDepth) {
      this.depthTarget = new THREE.WebGLRenderTarget(
        this.depthCaptureSize.width,
        this.depthCaptureSize.height,
      );
      this.depthTarget.texture.minFilter = THREE.NearestFilter;
      this.depthTarget.texture.magFilter = THREE.NearestFilter;
      this.depthTarget.texture.generateMipmaps = false;
      this.depthTarget.depthTexture = new THREE.DepthTexture(
        this.depthCaptureSize.width,
        this.depthCaptureSize.height,
      );
      this.depthTarget.depthTexture.format = THREE.DepthFormat;
      this.depthTarget.depthTexture.type = THREE.FloatType;

      this.depthInferenceMaterial = new THREE.ShaderMaterial({
        uniforms: {
          tDepth: { value: this.depthTarget.depthTexture },
          cameraNear: { value: this.camera.near },
          cameraFar: { value: this.camera.far },
        },
        vertexShader: `
          varying vec2 vUv;
          void main() {
            vUv = uv;
            gl_Position = vec4(position.xy, 0.0, 1.0);
          }
        `,
        fragmentShader: `
          #include <packing>
          uniform sampler2D tDepth;
          uniform float cameraNear;
          uniform float cameraFar;
          varying vec2 vUv;
          void main() {
            float depth = texture2D(tDepth, vUv).x;
            float viewZ = perspectiveDepthToViewZ(depth, cameraNear, cameraFar);
            float linearDepth = -viewZ;
            gl_FragColor = vec4(linearDepth, 0.0, 0.0, 1.0);
          }
        `,
      });
      this.depthInferenceScene = new THREE.Scene();
      this.depthInferenceScene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), this.depthInferenceMaterial));
      this.depthOrthoCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
      this.depthInferenceTarget = new THREE.WebGLRenderTarget(
        this.depthCaptureSize.width,
        this.depthCaptureSize.height,
        {
          type: THREE.FloatType,
          format: THREE.RGBAFormat,
          minFilter: THREE.NearestFilter,
          magFilter: THREE.NearestFilter,
          depthBuffer: false,
          stencilBuffer: false,
        },
      );
      this.depthPixels = new Float32Array(this.depthCaptureSize.width * this.depthCaptureSize.height * 4);
      this.depthFrame = new Float32Array(this.depthCaptureSize.width * this.depthCaptureSize.height);
    }

    const resizeObserver = new ResizeObserver(() => {
      const w = container.clientWidth;
      const h = container.clientHeight;
      this.renderer.setSize(w, h);
      if (!this.isDepth) {
        this.camera.aspect = w / h;
        this.camera.updateProjectionMatrix();
      }
    });
    resizeObserver.observe(container);
  }

  updateCamera(headPos: THREE.Vector3, headQuat: THREE.Quaternion, offset = new THREE.Vector3(0, 0, 0.08)) {
    const forward = offset.clone().applyQuaternion(headQuat);
    this.camera.position.copy(headPos).add(forward);
    this.camera.quaternion.copy(headQuat);
    this.camera.rotateX(-0.08); // slight downward tilt
  }

  updatePose(position: THREE.Vector3, quaternion: THREE.Quaternion) {
    this.camera.position.copy(position);
    this.camera.quaternion.copy(quaternion);
  }

  captureDepth(scene: THREE.Scene) {
    if (
      !this.isDepth ||
      !this.depthTarget ||
      !this.depthInferenceTarget ||
      !this.depthInferenceMaterial ||
      !this.depthInferenceScene ||
      !this.depthOrthoCamera ||
      !this.depthPixels ||
      !this.depthFrame
    ) {
      return null;
    }

    this.depthInferenceMaterial.uniforms.cameraNear.value = this.camera.near;
    this.depthInferenceMaterial.uniforms.cameraFar.value = this.camera.far;
    try {
      this.renderer.setRenderTarget(this.depthTarget);
      this.renderer.clear();
      this.renderer.render(scene, this.camera);

      this.renderer.setRenderTarget(this.depthInferenceTarget);
      this.renderer.clear();
      this.renderer.render(this.depthInferenceScene, this.depthOrthoCamera);
      this.renderer.readRenderTargetPixels(
        this.depthInferenceTarget,
        0,
        0,
        this.depthCaptureSize.width,
        this.depthCaptureSize.height,
        this.depthPixels,
      );
      this.renderer.setRenderTarget(null);
    } catch (error) {
      this.renderer.setRenderTarget(null);
      if (!this.depthCaptureWarned) {
        console.warn('Depth capture failed; policy will use zero depth features:', error);
        this.depthCaptureWarned = true;
      }
      return null;
    }

    for (let i = 0; i < this.depthFrame.length; i++) {
      this.depthFrame[i] = this.depthPixels[i * 4];
    }
    return {
      data: this.depthFrame,
      width: this.depthCaptureSize.width,
      height: this.depthCaptureSize.height,
    };
  }

  render(scene: THREE.Scene, background?: THREE.Color | THREE.Texture) {
    if (this.isDepth) {
      this.savedBackground = scene.background as THREE.Color | null;
      scene.background = new THREE.Color(0xffffff);
      this.depthMaterial.uniforms.near.value = this.camera.near;
      this.depthMaterial.uniforms.far.value = this.camera.far;
      const originalOverride = scene.overrideMaterial;
      scene.overrideMaterial = this.depthMaterial;
      this.renderer.render(scene, this.camera);
      scene.overrideMaterial = originalOverride;
      scene.background = this.savedBackground;
    } else {
      this.renderer.render(scene, this.camera);
    }
  }
}
