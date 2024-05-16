// eslint-disable-next-line no-unused-vars
import React from "react";
import { Pending } from "@mui/icons-material";

// eslint-disable-next-line react/prop-types
export default function PredictionResult({ prediction }) {
  return (
    <div className="flex gap-10 font-bold text-3xl">
      <a className="fade-in">Predicted label: </a>
      <a className="fade-in-slow anim-delay-075">{prediction || <Pending />}</a>
    </div>
  );
}
