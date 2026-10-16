import { NextRequest, NextResponse } from "next/server";

export const runtime = "edge";

/**
 * Server-side audio transcription proxy using Groq Cloud Whisper-large-v3.
 * Protects GROQ_API_KEY from exposure to client browsers.
 */
export async function POST(req: NextRequest) {
  try {
    const apiKey = process.env.GROQ_API_KEY;
    if (!apiKey) {
      return NextResponse.json(
        { error: "GROQ_API_KEY is not configured on the server." },
        { status: 500 }
      );
    }

    const formData = await req.formData();
    const file = formData.get("file") || formData.get("audio");

    if (!file || !(file instanceof Blob)) {
      return NextResponse.json(
        { error: "Audio file blob is required in form-data payload." },
        { status: 400 }
      );
    }

    // Prepare outbound multipart request for Groq
    const groqPayload = new FormData();
    groqPayload.append("file", file, "audio.webm");
    groqPayload.append("model", "whisper-large-v3");
    groqPayload.append("response_format", "json");
    groqPayload.append("temperature", "0.0");

    const tStart = performance.now();
    const res = await fetch("https://api.groq.com/openai/v1/audio/transcriptions", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${apiKey}`,
      },
      body: groqPayload,
    });

    const elapsedMs = Math.round(performance.now() - tStart);

    if (!res.ok) {
      const errText = await res.text();
      return NextResponse.json(
        { error: `Groq STT failed (${res.status}): ${errText}` },
        { status: res.status }
      );
    }

    const data = await res.json();
    return NextResponse.json({
      text: data.text?.trim() || "",
      duration_ms: elapsedMs,
      provider: "groq-whisper-large-v3",
    });
  } catch (err: any) {
    return NextResponse.json(
      { error: `Internal transcription proxy error: ${err.message}` },
      { status: 500 }
    );
  }
}
