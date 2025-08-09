"use client";
import { useEffect, useRef, useState } from "react";

// Example JSON format (replace with real data)
const matchData = {
  "frame_1": {
    player_positions: {
      player1: [50, 100],
      player2: [150, 80],
      player3: [250, 150]
    }
  },
  "frame_2": {
    player_positions: {
      player1: [60, 110],
      player2: [160, 85],
      player3: [260, 155]
    }
  }
  // more frames...
};

export function VolleyballCourt() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [frameIndex, setFrameIndex] = useState(0);
  const frameKeys = Object.keys(matchData);

  const drawCourt = (ctx: CanvasRenderingContext2D) => {
    ctx.fillStyle = "#f5deb3"; // court color
    ctx.fillRect(0, 0, ctx.canvas.width, ctx.canvas.height);

    // center line
    ctx.strokeStyle = "#000";
    ctx.beginPath();
    ctx.moveTo(ctx.canvas.width / 2, 0);
    ctx.lineTo(ctx.canvas.width / 2, ctx.canvas.height);
    ctx.stroke();
  };

  const drawPlayers = (ctx: CanvasRenderingContext2D, positions: Record<string, [number, number]>) => {
    ctx.fillStyle = "blue";
    Object.values(positions).forEach(([x, y]) => {
      ctx.beginPath();
      ctx.arc(x, y, 10, 0, Math.PI * 2); // radius = 10px
      ctx.fill();
    });
  };

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    drawCourt(ctx);
    drawPlayers(ctx, matchData[frameKeys[frameIndex]].player_positions);
  }, [frameIndex, frameKeys]);

  // Loop frames automatically
  useEffect(() => {
    const interval = setInterval(() => {
      setFrameIndex((prev) => (prev + 1) % frameKeys.length);
    }, 500); // every 0.5s
    return () => clearInterval(interval);
  }, [frameKeys.length]);

  return (
    <div style={{ textAlign: "center" }}>
      <canvas ref={canvasRef} width={600} height={300} style={{ border: "2px solid black" }} />
      <div>
        <button onClick={() => setFrameIndex((prev) => (prev + 1) % frameKeys.length)}>
          Next Frame
        </button>
      </div>
    </div>
  );
}

