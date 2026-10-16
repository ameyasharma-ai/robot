"use client";

import React, { useEffect, useRef, useState } from "react";

interface Message {
  role: "user" | "assistant";
  content: string;
}

export default function CyberdeckWebPage() {
  const [messages, setMessages] = useState<Message[]>([
    {
      role: "assistant",
      content:
        "Robot Core Cyberdeck online. Edge pipeline initialized with Groq Cloud Whisper and Gemini 2.0 streaming. Press and hold Speak or type below.",
    },
  ]);
  const [input, setInput] = useState("");
  const [persona, setPersona] = useState("cypher");
  const [status, setStatus] = useState<"idle" | "listening" | "thinking" | "speaking">("idle");
  const [isRecording, setIsRecording] = useState(false);

  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const animFrameRef = useRef<number | null>(null);
  const transcriptEndRef = useRef<HTMLDivElement | null>(null);

  // Auto-scroll transcript viewport
  useEffect(() => {
    transcriptEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Waveform Visualizer
  const drawWaveform = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const analyser = analyserRef.current;
    const bufferLength = analyser ? analyser.frequencyBinCount : 64;
    const dataArray = new Uint8Array(bufferLength);

    const render = () => {
      animFrameRef.current = requestAnimationFrame(render);
      const width = canvas.width;
      const height = canvas.height;

      ctx.clearRect(0, 0, width, height);

      if (analyser && isRecording) {
        analyser.getByteFrequencyData(dataArray);
      } else {
        // Idle gentle wave animation
        const t = performance.now() * 0.003;
        for (let i = 0; i < bufferLength; i++) {
          dataArray[i] = Math.sin(t + i * 0.15) * 20 + 25;
        }
      }

      ctx.lineWidth = 2;
      const gradient = ctx.createLinearGradient(0, 0, width, 0);
      gradient.addColorStop(0, "#00f0ff");
      gradient.addColorStop(0.5, "#00ff9d");
      gradient.addColorStop(1, "#00f0ff");
      ctx.strokeStyle = gradient;

      ctx.beginPath();
      const sliceWidth = width / bufferLength;
      let x = 0;

      for (let i = 0; i < bufferLength; i++) {
        const v = dataArray[i] / 128.0;
        const y = (v * height) / 2.5;

        if (i === 0) {
          ctx.moveTo(x, height / 2 - y / 2);
        } else {
          ctx.lineTo(x, height / 2 - y / 2);
        }
        x += sliceWidth;
      }

      ctx.stroke();
    };

    render();
  };

  useEffect(() => {
    drawWaveform();
    return () => {
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
    };
  }, [isRecording]);

  // Voice Recording via Web Audio API
  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      audioChunksRef.current = [];

      const audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)();
      const source = audioCtx.createMediaStreamSource(stream);
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 128;
      source.connect(analyser);

      audioCtxRef.current = audioCtx;
      analyserRef.current = analyser;

      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) audioChunksRef.current.push(e.data);
      };

      mediaRecorder.onstop = async () => {
        stream.getTracks().forEach((track) => track.stop());
        const audioBlob = new Blob(audioChunksRef.current, { type: "audio/webm" });
        await handleAudioTranscription(audioBlob);
      };

      mediaRecorder.start();
      mediaRecorderRef.current = mediaRecorder;
      setIsRecording(true);
      setStatus("listening");
    } catch (err: any) {
      console.error("Microphone access failed:", err);
      setStatus("idle");
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  const handleAudioTranscription = async (blob: Blob) => {
    setStatus("thinking");
    try {
      const fd = new FormData();
      fd.append("file", blob, "audio.webm");

      const res = await fetch("/api/voice/transcribe", {
        method: "POST",
        body: fd,
      });

      if (!res.ok) throw new Error("Transcription failed");
      const data = await res.json();
      if (data.text) {
        await sendMessage(data.text);
      } else {
        setStatus("idle");
      }
    } catch (err) {
      console.error("STT Error:", err);
      setStatus("idle");
    }
  };

  // Streaming Text & Audio Synthesis
  const sendMessage = async (textToSend: string) => {
    const q = textToSend.trim();
    if (!q) return;

    setInput("");
    const newMessages: Message[] = [...messages, { role: "user", content: q }];
    setMessages(newMessages);
    setStatus("thinking");

    try {
      const res = await fetch("/api/chat/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messages: newMessages, persona }),
      });

      if (!res.ok) throw new Error("Chat streaming failed");
      if (!res.body) throw new Error("No response stream");

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let assistantReply = "";

      setMessages((prev) => [...prev, { role: "assistant", content: "" }]);
      setStatus("speaking");

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split("\n");

        for (const line of lines) {
          if (line.startsWith("data: ")) {
            const dataStr = line.slice(6).trim();
            if (dataStr === "[DONE]") break;
            try {
              const parsed = JSON.parse(dataStr);
              if (parsed.text) {
                assistantReply += parsed.text;
                setMessages((prev) => {
                  const updated = [...prev];
                  updated[updated.length - 1] = {
                    role: "assistant",
                    content: assistantReply,
                  };
                  return updated;
                });
              }
            } catch {
              // Ignore partial parse
            }
          }
        }
      }

      // Synthesize audio response via Edge TTS
      if (assistantReply) {
        playTTS(assistantReply);
      } else {
        setStatus("idle");
      }
    } catch (err) {
      console.error("Chat error:", err);
      setStatus("idle");
    }
  };

  const playTTS = async (text: string) => {
    try {
      const res = await fetch("/api/voice/synthesize", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, voice: "en-US-AvaNeural" }),
      });

      if (res.headers.get("Content-Type")?.includes("audio/mpeg")) {
        const audioBlob = await res.blob();
        const url = URL.createObjectURL(audioBlob);
        const audio = new Audio(url);
        audio.onended = () => {
          URL.revokeObjectURL(url);
          setStatus("idle");
        };
        audio.onerror = () => {
          URL.revokeObjectURL(url);
          setStatus("idle");
        };
        audio.play().catch((err) => {
          console.warn("Autoplay deferred or blocked by browser policy:", err);
          URL.revokeObjectURL(url);
          setStatus("idle");
        });
      } else {
        // Web Speech API fallback
        if ("speechSynthesis" in window) {
          const utterance = new SpeechSynthesisUtterance(text);
          utterance.onend = () => setStatus("idle");
          utterance.onerror = () => setStatus("idle");
          window.speechSynthesis.speak(utterance);
        } else {
          setStatus("idle");
        }
      }
    } catch {
      setStatus("idle");
    }
  };

  return (
    <div className="cyber-container">
      {/* Header */}
      <header className="cyber-header">
        <div className="brand-section">
          <div className="brand-logo">⚡</div>
          <div>
            <div className="brand-title">Robot Core</div>
            <div style={{ display: "flex", gap: "8px", alignItems: "center" }}>
              <span className="brand-badge">WEB-V2.0-EDGE</span>
              <span style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>Groq + Gemini Cloud</span>
            </div>
          </div>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
          {/* Persona selector */}
          <select
            value={persona}
            onChange={(e) => setPersona(e.target.value)}
            style={{
              background: "rgba(0,0,0,0.6)",
              color: "var(--text-main)",
              border: "1px solid var(--border-subtle)",
              padding: "6px 12px",
              borderRadius: "var(--radius-sm)",
              fontFamily: "var(--font-display)",
              fontSize: "0.85rem",
              textTransform: "uppercase",
              outline: "none",
            }}
          >
            <option value="cypher">Cypher (Tactical)</option>
            <option value="aria">Aria (Empathetic)</option>
            <option value="luna">Luna (Hacker)</option>
            <option value="glados">GLaDOS (Analytical)</option>
          </select>

          {/* Status Pill */}
          <div className={`status-pill ${status}`}>
            <span className="status-dot" />
            <span>{status}</span>
          </div>
        </div>
      </header>

      {/* Visualizer Waveform */}
      <section className="visualizer-deck">
        <canvas ref={canvasRef} className="waveform-canvas" width={800} height={72} />
      </section>

      {/* Transcript Viewport */}
      <main className="transcript-viewport">
        {messages.map((m, idx) => (
          <div key={idx} className={`chat-bubble ${m.role}`}>
            <div className="chat-sender">{m.role === "assistant" ? persona : "Operator"}</div>
            <div>{m.content}</div>
          </div>
        ))}
        <div ref={transcriptEndRef} />
      </main>

      {/* Control Dock */}
      <footer className="control-dock">
        <button
          className={`cyber-btn mic ${isRecording ? "active" : ""}`}
          onMouseDown={startRecording}
          onMouseUp={stopRecording}
          onTouchStart={startRecording}
          onTouchEnd={stopRecording}
          title="Push to Talk (hold to speak)"
        >
          🎙️
        </button>

        <input
          type="text"
          className="cyber-input"
          placeholder="Transmit directive or spoken query..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              sendMessage(input);
            }
          }}
        />

        <button className="cyber-btn primary" onClick={() => sendMessage(input)}>
          Transmit
        </button>
      </footer>
    </div>
  );
}
