"use client";
import React from "react";
import { Box } from "@mui/material";

interface VideoPlayerProps {
  src: string; // path relative to public folder, e.g. "/assets/test.mp4"
  title?: string;
}

export function VideoPlayer({ src, title = "Video" }: VideoPlayerProps) {
  return (
    <Box
      sx={{
        position: "relative",
        paddingTop: "56.25%", // 16:9 aspect ratio
        width: "100%",
        borderRadius: 2,
        overflow: "hidden",
        boxShadow: 3,
      }}
    >
      <video
        src={src}
        title={title}
        controls
        muted
        loop
        style={{
          position: "absolute",
          top: 0,
          left: 0,
          width: "100%",
          height: "100%",
          objectFit: "cover",
        }}
      />
    </Box>
  );
}
