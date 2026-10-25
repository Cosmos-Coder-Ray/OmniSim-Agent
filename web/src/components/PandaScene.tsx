import { useFrame, useThree } from "@react-three/fiber";
import { useEffect, useMemo, useRef, type MutableRefObject } from "react";
import * as THREE from "three";
import type { Replay } from "../api/types";
import { heatColor } from "../replay/geometry";

interface PandaSceneProps {
  replay: Replay;
  nominal: Replay | null;
  timeRef: MutableRefObject<number>;
  selectedJointRef: MutableRefObject<number | null>;
  onReady: (seek: (time: number) => void) => void;
  onReadout: (ratios: Float32Array, joint: number | null) => void;
}

function frameAt(replay: Replay, time: number): [number, number, number] {
  const frames = replay.frames;
  let low = 0;
  let high = frames.length - 1;
  const t = Math.max(frames[0].t, Math.min(frames[high].t, time));
  while (high - low > 1) {
    const middle = (low + high) >> 1;
    if (frames[middle].t <= t) low = middle;
    else high = middle;
  }
  const portion = high === low ? 0 : (t - frames[low].t) / (frames[high].t - frames[low].t);
  return [low, high, portion];
}

function lineGeometry(pointCount: number): THREE.BufferGeometry {
  const geometry = new THREE.BufferGeometry();
  const positions = new THREE.BufferAttribute(new Float32Array(pointCount * 3), 3);
  positions.setUsage(THREE.DynamicDrawUsage);
  geometry.setAttribute("position", positions);
  return geometry;
}

export function PandaScene({
  replay,
  nominal,
  timeRef,
  selectedJointRef,
  onReady,
  onReadout,
}: PandaSceneProps) {
  const spheres = useRef<THREE.InstancedMesh>(null);
  const eeMarker = useRef<THREE.Mesh>(null);
  const { invalidate } = useThree();
  const ratiosRef = useRef(new Float32Array(7));
  const arm = useMemo(() => new THREE.Line(
    lineGeometry(8), new THREE.LineBasicMaterial({ color: "#d7e1e7" }),
  ), []);
  const ghost = useMemo(() => new THREE.Line(
    lineGeometry(8), new THREE.LineBasicMaterial({ color: "#91a4b2", transparent: true, opacity: 0.42 }),
  ), []);
  const trail = useMemo(() => new THREE.Line(
    lineGeometry(400), new THREE.LineBasicMaterial({ color: "#71d6bd", transparent: true, opacity: 0.55 }),
  ), []);
  const matrix = useMemo(() => new THREE.Matrix4(), []);
  const quaternion = useMemo(() => new THREE.Quaternion(), []);
  const position = useMemo(() => new THREE.Vector3(), []);
  const scale = useMemo(() => new THREE.Vector3(1, 1, 1), []);
  const color = useMemo(() => new THREE.Color(), []);

  useEffect(() => {
    onReady((nextTime) => {
      timeRef.current = Math.max(0, nextTime);
      invalidate();
    });
    return () => {
      arm.geometry.dispose();
      (arm.material as THREE.Material).dispose();
      ghost.geometry.dispose();
      (ghost.material as THREE.Material).dispose();
      trail.geometry.dispose();
      (trail.material as THREE.Material).dispose();
    };
  }, [arm, ghost, invalidate, onReady, timeRef, trail]);

  useFrame(() => {
    const mesh = spheres.current;
    const marker = eeMarker.current;
    if (!mesh || !marker) return;

    const [low, high, portion] = frameAt(replay, timeRef.current);
    const first = replay.frames[low];
    const second = replay.frames[high];
    const points = arm.geometry.getAttribute("position") as THREE.BufferAttribute;
    for (let index = 0; index < 8; index += 1) {
      const a = first.links[index];
      const b = second.links[index];
      const x = a[0] + (b[0] - a[0]) * portion;
      const y = a[1] + (b[1] - a[1]) * portion;
      const z = a[2] + (b[2] - a[2]) * portion;
      points.setXYZ(index, x, y, z);
      position.set(x, y, z);
      matrix.compose(position, quaternion, scale);
      mesh.setMatrixAt(index, matrix);
      const ratio = index === 0 ? null : first.tau_ratio[index - 1] +
        (second.tau_ratio[index - 1] - first.tau_ratio[index - 1]) * portion;
      if (ratio === null) color.set("#bac4ca");
      else {
        ratiosRef.current[index - 1] = ratio;
        color.set(heatColor(ratio));
      }
      mesh.setColorAt(index, color);
    }
    points.needsUpdate = true;
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    arm.geometry.getAttribute("position").needsUpdate = true;

    const eeX = first.ee[0] + (second.ee[0] - first.ee[0]) * portion;
    const eeY = first.ee[1] + (second.ee[1] - first.ee[1]) * portion;
    const eeZ = first.ee[2] + (second.ee[2] - first.ee[2]) * portion;
    marker.position.set(eeX, eeY, eeZ);

    const trailPosition = trail.geometry.getAttribute("position") as THREE.BufferAttribute;
    const trailStart = Math.max(0, low - 399);
    const trailCount = low - trailStart + 1;
    for (let index = 0; index < trailCount; index += 1) {
      const ee = replay.frames[trailStart + index].ee;
      trailPosition.setXYZ(index, ee[0], ee[1], ee[2]);
    }
    trailPosition.needsUpdate = true;
    trail.geometry.setDrawRange(0, trailCount);

    if (nominal) {
      const [ghostLow, ghostHigh, ghostPortion] = frameAt(nominal, timeRef.current);
      const ghostFirst = nominal.frames[ghostLow];
      const ghostSecond = nominal.frames[ghostHigh];
      const ghostPosition = ghost.geometry.getAttribute("position") as THREE.BufferAttribute;
      for (let index = 0; index < 8; index += 1) {
        const a = ghostFirst.links[index];
        const b = ghostSecond.links[index];
        ghostPosition.setXYZ(
          index,
          a[0] + (b[0] - a[0]) * ghostPortion,
          a[1] + (b[1] - a[1]) * ghostPortion,
          a[2] + (b[2] - a[2]) * ghostPortion,
        );
      }
      ghostPosition.needsUpdate = true;
      ghost.geometry.setDrawRange(0, 8);
    } else {
      ghost.geometry.setDrawRange(0, 0);
    }

    const joint = selectedJointRef.current;
    onReadout(ratiosRef.current, joint);
  });

  function selectJoint(instanceId: number | undefined) {
    if (instanceId === undefined) return;
    selectedJointRef.current = instanceId === 0 ? null : instanceId - 1;
    invalidate();
  }

  return (
    <>
      <primitive object={trail} frustumCulled={false} />
      <primitive object={ghost} frustumCulled={false} />
      <primitive object={arm} frustumCulled={false} />
      <instancedMesh
        ref={spheres}
        args={[undefined, undefined, 8]}
        frustumCulled={false}
        onClick={(event) => selectJoint(event.instanceId)}
      >
        <sphereGeometry args={[0.032, 16, 12]} />
        <meshStandardMaterial roughness={0.36} metalness={0.08} />
      </instancedMesh>
      <mesh ref={eeMarker}>
        <sphereGeometry args={[0.018, 12, 10]} />
        <meshStandardMaterial color="#f6c36a" emissive="#4b3214" roughness={0.32} />
      </mesh>
    </>
  );
}
