import { Grid, OrbitControls, Stats } from "@react-three/drei";
import { Canvas } from "@react-three/fiber";
import { forwardRef, useCallback, useImperativeHandle, useRef } from "react";
import type { Replay } from "../api/types";
import { jointLabel } from "../replay/geometry";
import { PandaScene } from "./PandaScene";

export interface ReplayCanvasHandle {
  setTime: (time: number) => void;
}

interface ReplayCanvasProps {
  replay: Replay;
  nominal: Replay | null;
  initialTime: number;
}

const debugStats = import.meta.env.DEV && new URLSearchParams(window.location.search).get("debug") === "1";

export const ReplayCanvas = forwardRef<ReplayCanvasHandle, ReplayCanvasProps>(
  function ReplayCanvas({ replay, nominal, initialTime }, ref) {
    const time = useRef(initialTime);
    const firstRatios = replay.frames[0].tau_ratio;
    const maxRatio = Math.max(...firstRatios);
    const selectedJoint = useRef<number | null>(firstRatios.indexOf(maxRatio));
    const seek = useRef<(time: number) => void>(() => undefined);
    const readout = useRef<HTMLSpanElement>(null);

    const setTime = useCallback((nextTime: number) => seek.current(nextTime), []);
    const onReady = useCallback((setSceneTime: (time: number) => void) => {
      seek.current = setSceneTime;
      setSceneTime(time.current);
    }, []);
    const onReadout = useCallback((ratios: Float32Array, joint: number | null) => {
      if (!readout.current) return;
      if (joint === null) {
        readout.current.textContent = "GLOBAL · BASE LINK";
        return;
      }
      readout.current.textContent = `${jointLabel(joint)} · TORQUE ${(ratios[joint] * 100).toFixed(0)}%`;
    }, []);

    useImperativeHandle(ref, () => ({ setTime }), [setTime]);

    return (
      <div className="replay-scene-layout">
        <div className="canvas-frame" aria-label="Panda replay canvas">
          <Canvas
            frameloop="demand"
            dpr={[1, 1.5]}
            camera={{ position: [1.3, 1, 1.3], fov: 50 }}
            gl={{ antialias: true, alpha: false }}
          >
            <color attach="background" args={["#0d141a"]} />
            <ambientLight intensity={1.25} />
            <directionalLight position={[1.2, 2.2, 1.5]} intensity={1.7} />
            <Grid
              position={[0, -0.005, 0]}
              args={[2, 2]}
              cellSize={0.1}
              sectionSize={0.5}
              cellColor="#20303a"
              sectionColor="#344852"
              fadeDistance={3}
            />
            <group rotation={[-Math.PI / 2, 0, 0]}>
              <PandaScene
                replay={replay}
                nominal={nominal}
                timeRef={time}
                selectedJointRef={selectedJoint}
                onReady={onReady}
                onReadout={onReadout}
              />
            </group>
            <OrbitControls enableDamping={false} minDistance={0.65} maxDistance={3.5} />
            {debugStats && <Stats showPanel={0} />}
          </Canvas>
        </div>
        <aside className="scene-aside">
          <div className="heat-legend" aria-label="Torque ratio heat legend">
            <div className="section-kicker">TORQUE RATIO</div>
            <div className="heat-legend-bar" />
            <div className="heat-legend-labels"><span>0.0</span><span>0.55</span><span>1.2+</span></div>
          </div>
          <div className="joint-readout">
            <div className="section-kicker">SELECTED JOINT</div>
            <span ref={readout} aria-live="polite" data-testid="joint-readout">Select a link</span>
          </div>
          {nominal && <span className="ghost-key"><i /> Nominal reference</span>}
        </aside>
      </div>
    );
  },
);
