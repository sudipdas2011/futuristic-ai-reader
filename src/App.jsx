import React, { useRef, useState, useEffect } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";

import { initializeReader, predictCanvas } from "./ai/reader";

export default function App() {
  const canvasRef = useRef(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [confidences, setConfidences] = useState(Array(10).fill(0));

  // Custom Controls State
  const [brushSize, setBrushSize] = useState(16);
  const [brushColor, setBrushColor] = useState("#18181b"); 

  // Global Mouse Tracking State for the Custom Follower Dot
  const [mousePos, setMousePos] = useState({ x: -100, y: -100 });

  useEffect(() => {
    initializeReader();

    const handleGlobalMouseMove = (e) => {
      setMousePos({
        x: e.clientX,
        y: e.clientY
      });
    };

    window.addEventListener(
      "mousemove",
      handleGlobalMouseMove
    );

    return () => {
      window.removeEventListener(
        "mousemove",
        handleGlobalMouseMove
      );
    };
  }, []);

  // --- Dynamic High-DPI Spatial Coordinates Vector Math ---
  const getCanvasCoords = (e) => {
    const canvas = canvasRef.current;
    if (!canvas) return { x: 0, y: 0 };
    const rect = canvas.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * canvas.width;
    const y = ((e.clientY - rect.top) / rect.height) * canvas.height;
    return { x, y };
  };

  const startDrawing = (e) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const { x, y } = getCanvasCoords(e);
    ctx.beginPath();
    ctx.moveTo(x, y);
    setIsDrawing(true);
  };

  const draw = (e) => {
    if (!isDrawing) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const { x, y } = getCanvasCoords(e);
    ctx.lineTo(x, y);
    ctx.strokeStyle = brushColor; 
    ctx.lineWidth = brushSize; 
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.stroke();
  };

  const stopDrawing = () => {
    if (!isDrawing) return;
    setIsDrawing(false);
    recognizeDigit();
  };

  const clearCanvas = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    setPrediction(null);
    setConfidences(Array(10).fill(0));
  };

  // --- Real-Time Neural Signal Array Operations ---
  const recognizeDigit = async () => {
    if (!canvasRef.current) {
      return;
    }

    try {
      const result = predictCanvas( canvasRef.current);

      setPrediction( result.prediction.toString());

      setConfidences(resultconfidences);
    } catch (error) {
      console.error( "AI prediction failed:", error );
    }
  };

  return (
    <div className="flex h-screen items-center justify-center bg-stone-50 p-4 font-mono text-zinc-800 select-none relative overflow-hidden">
      
      {/* GLOBAL FLOATING CURSOR FOLLOWER DOT */}
      <div 
        className="fixed rounded-full pointer-events-none z-50 transition-all duration-75 ease-out -translate-x-1/2 -translate-y-1/2 opacity-80"
        style={{ 
          left: `${mousePos.x}px`, 
          top: `${mousePos.y}px`,
          width: `${brushSize}px`,
          height: `${brushSize}px`,
          backgroundColor: brushColor 
        }}
      />

      <Card className="w-full max-w-4xl max-h-[95vh] border-stone-200 bg-white/80 shadow-xl backdrop-blur-md rounded-2xl z-10 overflow-hidden flex flex-col">
        <CardContent className="flex flex-col gap-4 p-4 h-full overflow-hidden justify-between">
          
          {/* TOP MODULE: SLATE CANVAS AREA */}
          <div className="flex flex-col items-center border border-stone-200/80 rounded-xl p-3 bg-stone-100/40 w-full shrink-0">
            <canvas
              ref={canvasRef}
              width={800}
              height={260} // Reduced height slightly to strictly fit within 100vh bounds
              onMouseDown={startDrawing}
              onMouseMove={draw}
              onMouseUp={stopDrawing}
              onMouseLeave={stopDrawing}
              className="w-full cursor-none rounded-lg border border-stone-200 bg-stone-50 shadow-inner transition-all duration-300 hover:border-stone-400 aspect-[800/260]"
            />
            
            {/* UTILITY CONTROL HUB */}
            <div className="mt-3 flex flex-col md:flex-row items-center justify-between gap-4 w-full border-t border-stone-200/60 pt-3">
              <div className="flex flex-wrap items-center gap-6 w-full md:w-auto">
                {/* Size Slider */}
                <div className="flex items-center gap-3 min-w-[180px] flex-1 sm:flex-initial">
                  <span className="text-xs font-bold text-zinc-500 whitespace-nowrap">SIZE: {brushSize}px</span>
                  <input
                    type="range"
                    min="4"
                    max="40"
                    value={brushSize}
                    onChange={(e) => setBrushSize(parseInt(e.target.value))}
                    className="w-full h-1 bg-stone-200 rounded-lg appearance-none cursor-pointer accent-zinc-900"
                  />
                </div>

                {/* Color Selector */}
                <div className="flex items-center gap-2">
                  <span className="text-xs font-bold text-zinc-500 mr-1">INK:</span>
                  {[
                    { name: "Charcoal", value: "#18181b" },
                    { name: "Slate Blue", value: "#3b82f6" },
                    { name: "Crimson", value: "#ef4444" },
                    { name: "Emerald", value: "#10b981" }
                  ].map((color) => (
                    <button
                      key={color.value}
                      onClick={() => brushColor !== color.value && setBrushColor(color.value)}
                      className={`w-5 h-5 rounded-full border transition-all duration-150 ${
                        brushColor === color.value 
                          ? "ring-2 ring-offset-2 ring-zinc-900 scale-110" 
                          : "border-stone-300 hover:scale-105"
                      }`}
                      style={{ backgroundColor: color.value }}
                      title={color.name}
                    />
                  ))}
                </div>
              </div>

              <Button 
                onClick={clearCanvas} 
                variant="outline" 
                className="w-full md:w-40 border-stone-200 bg-white text-zinc-700 hover:bg-stone-100 hover:text-red-600 font-bold tracking-widest text-xs h-8 shadow-sm transition-all shrink-0"
              >
                RESET SLATE
              </Button>
            </div>
          </div>

          {/* BOTTOM MODULE: SYMMETRIC ANALYTICS PANELS */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 w-full flex-1 overflow-hidden min-h-0">
            
            {/* ANALYTICS LEFT: DIGITAL READOUT */}
            <div className="flex flex-col items-center justify-center border border-stone-200/80 rounded-xl p-4 bg-stone-100/40 h-full overflow-hidden">
              <div className="text-zinc-400 text-[10px] font-bold tracking-widest mb-1 uppercase">CORE READOUT</div>
              <div className="text-8xl font-light tracking-tighter text-zinc-900 font-mono transition-all duration-300 leading-none">
                {prediction !== null ? prediction : "—"}
              </div>
            </div>

            {/* ANALYTICS RIGHT: CONFIDENCE LEDGER */}
            <div className="border border-stone-200/80 rounded-xl p-4 bg-stone-100/40 h-full flex flex-col justify-between overflow-hidden">
              <div className="text-zinc-400 text-[10px] font-bold tracking-widest mb-2 text-center uppercase">RADAR DISTRIBUTION MATRIX</div>
              <div className="flex flex-col gap-1.5 flex-1 justify-center overflow-y-auto">
                {confidences.map((prob, idx) => {
                  const percentage = Math.round(prob * 100);
                  const isMatch = prediction === idx.toString() && percentage > 0;
                  
                  return (
                    <div key={idx} className="flex items-center gap-2 w-full text-[11px] leading-none">
                      <span className={`w-3 font-mono font-bold ${isMatch ? "text-zinc-900" : "text-stone-400"}`}>{idx}</span>
                      <div className="flex-1 h-1 bg-stone-200/60 rounded-full overflow-hidden">
                        <div 
                          className={`h-full transition-all duration-300 rounded-full ${isMatch ? "bg-zinc-900" : "bg-stone-400/60"}`}
                          style={{ width: `${percentage}%` }}
                        />
                      </div>
                      <span className={`w-7 text-right font-mono ${isMatch ? "text-zinc-900 font-bold" : "text-stone-400"}`}>
                        {percentage}%
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>

          </div>

        </CardContent>
      </Card>
    </div>
  );
}
