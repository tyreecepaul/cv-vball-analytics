"use client";
import { useEffect, useRef, useState } from "react";

const API_ENDPOINT = "http://localhost:8000/api/match-data"

// Example JSON format (replace with real data)
const matchData = {
  /*"frame_1": {
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
  }*/  // more frames...
};

export function VolleyballCourt() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [frameIndex, setFrameIndex] = useState(0);
  const [matchData, setMatchData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const frameKeys = matchData ? Object.keys(matchData) : [];

  // Fetch data from API
  useEffect(() => {
    const fetchMatchData = async () => {
      try {
        setLoading(true);
        const response = await fetch(API_ENDPOINT); // Replace with your API endpoint

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        setMatchData(data);
        setError(null);
      } catch (err) {
        console.error('Failed to fetch match data:', err);
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchMatchData();
  }, []);

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

  // Draw canvas when data changes
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !matchData) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    drawCourt(ctx);

    if (frameKeys.length > 0) {
      drawPlayers(ctx, matchData[frameKeys[frameIndex]].player_positions);
    }
  }, [frameIndex, frameKeys, matchData]);

  // Auto-advance frames
  useEffect(() => {
    if (!matchData || frameKeys.length === 0) return;

    const interval = setInterval(() => {
      setFrameIndex((prev) => (prev + 1) % frameKeys.length);
    }, 500); // every 0.5s

    return () => clearInterval(interval);
  }, [frameKeys.length, matchData]);

  // Loading state
  if (loading) {
    return (
      <div style={{ textAlign: "center", padding: "20px" }}>
        <div>Loading match data...</div>
      </div>
    );
  }

  // Error state
  if (error) {
    return (
      <div style={{ textAlign: "center", padding: "20px" }}>
        <div style={{ color: "red" }}>Error loading match data: {error}</div>
        <button onClick={() => window.location.reload()}>
          Retry
        </button>
      </div>
    );
  }

  // No data state
  if (!matchData) {
    return (
      <div style={{ textAlign: "center", padding: "20px" }}>
        <div>No match data available</div>
      </div>
    );
  }

  return (
    <div style={{ textAlign: "center" }}>
      <canvas
        ref={canvasRef}
        width={600}
        height={300}
        style={{ border: "2px solid black" }}
      />
      <div style={{ marginTop: "10px" }}>
        <button
          onClick={() => setFrameIndex((prev) => (prev + 1) % frameKeys.length)}
          disabled={frameKeys.length === 0}
        >
          Next Frame ({frameIndex + 1}/{frameKeys.length})
        </button>
      </div>
    </div>
  );
}

