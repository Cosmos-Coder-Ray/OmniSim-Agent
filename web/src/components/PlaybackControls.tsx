import { useEffect, useRef } from "react";
import type { MutableRefObject, RefObject } from "react";
import { playbackRate } from "../replay/motion";
import { useUrlStore } from "../state/urlStore";
import type { ReplayCanvasHandle } from "./ReplayCanvas";

interface PlaybackControlsProps {
  duration: number;
  canvas: RefObject<ReplayCanvasHandle | null>;
  onTime: (time: number) => void;
}

function formatTime(time: number): string {
  return `${time.toFixed(2)} s`;
}

function updateTime(
  time: number,
  duration: number,
  timeRef: MutableRefObject<number>,
  sliderRef: RefObject<HTMLInputElement | null>,
  labelRef: RefObject<HTMLOutputElement | null>,
  canvas: RefObject<ReplayCanvasHandle | null>,
  onTime: (time: number) => void,
) {
  const next = Math.max(0, Math.min(duration, time));
  timeRef.current = next;
  if (sliderRef.current) sliderRef.current.value = String(next);
  if (labelRef.current) labelRef.current.value = formatTime(next);
  canvas.current?.setTime(next);
  onTime(next);
}

export function PlaybackControls({ duration, canvas, onTime }: PlaybackControlsProps) {
  const currentTime = useUrlStore((state) => state.t);
  const isPlaying = useUrlStore((state) => state.isPlaying);
  const setTime = useUrlStore((state) => state.setT);
  const setPlaying = useUrlStore((state) => state.setPlaying);
  const seekTo = useUrlStore((state) => state.seekTo);
  const timeRef = useRef(Math.min(duration, currentTime));
  const sliderRef = useRef<HTMLInputElement>(null);
  const timeLabelRef = useRef<HTMLOutputElement>(null);

  useEffect(() => {
    if (!isPlaying) {
      updateTime(currentTime, duration, timeRef, sliderRef, timeLabelRef, canvas, onTime);
    }
  }, [canvas, currentTime, duration, isPlaying, onTime]);

  useEffect(() => {
    if (!isPlaying) return;
    let animationFrame = 0;
    let previousTime: number | undefined;
    const rate = playbackRate();
    const advance = (now: number) => {
      if (previousTime !== undefined) {
        updateTime(
          timeRef.current + ((now - previousTime) / 1000) * rate,
          duration,
          timeRef,
          sliderRef,
          timeLabelRef,
          canvas,
          onTime,
        );
      }
      previousTime = now;
      if (timeRef.current >= duration) {
        seekTo(duration);
        return;
      }
      animationFrame = window.requestAnimationFrame(advance);
    };
    animationFrame = window.requestAnimationFrame(advance);
    return () => window.cancelAnimationFrame(animationFrame);
  }, [canvas, duration, isPlaying, onTime, seekTo]);

  function togglePlayback() {
    if (isPlaying) {
      seekTo(timeRef.current);
      return;
    }
    if (timeRef.current >= duration) {
      updateTime(0, duration, timeRef, sliderRef, timeLabelRef, canvas, onTime);
      setTime(0);
    }
    setPlaying(true);
  }

  function seek(time: number) {
    seekTo(time);
    updateTime(time, duration, timeRef, sliderRef, timeLabelRef, canvas, onTime);
  }

  return (
    <section className="playback-controls" aria-label="Replay playback controls">
      <button className="button button-play" type="button" onClick={togglePlayback}>
        {isPlaying ? "Pause" : "Play"}
      </button>
      <label className="scrubber">
        <span className="visually-hidden">Seek replay time</span>
        <input
          ref={sliderRef}
          aria-label="Seek replay time"
          type="range"
          min={0}
          max={duration}
          step={0.01}
          value={currentTime}
          onChange={(event) => seek(Number(event.target.value))}
        />
      </label>
      <output className="playback-time" data-testid="playback-time" ref={timeLabelRef}>
        {formatTime(currentTime)}
      </output>
      {isPlaying && playbackRate() < 1 && <span className="motion-note">Reduced motion · 0.5×</span>}
    </section>
  );
}
