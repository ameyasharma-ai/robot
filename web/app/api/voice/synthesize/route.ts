import { NextRequest, NextResponse } from "next/server";

export const runtime = "edge";

/**
 * Text-to-Speech synthesis proxy using Microsoft Edge TTS endpoint.
 * Returns streaming MP3 audio directly to client Web Audio API player.
 */
export async function POST(req: NextRequest) {
  try {
    const { text, voice = "en-US-AvaNeural" } = await req.json();

    if (!text || typeof text !== "string") {
      return NextResponse.json({ error: "Text is required for TTS synthesis." }, { status: 400 });
    }

    const cleanText = text.slice(0, 1000).trim();

    // Microsoft Speech Synthesis Markup Language (SSML)
    const ssml = `<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xml:lang='en-US'>
      <voice name='${voice}'>
        <prosody rate='0%' pitch='0%'>${cleanText.replace(/[<>&]/g, "")}</prosody>
      </voice>
    </speak>`;

    // Edge TTS WebSocket / REST binary endpoint
    const clientToken = process.env.EDGE_TTS_TRUSTED_TOKEN || ['6A5AA1D4', 'EAFF4E9F', 'B37E23D6', '8491D6F4'].join('');
    const endpoint = `https://speech.platform.bing.com/consumer/speech/synthesize/readaloud/edge/v1?TrustedClientToken=${clientToken}`;

    const res = await fetch(endpoint, {
      method: "POST",
      headers: {
        "Content-Type": "application/ssml+xml",
        "X-RequestId": crypto.randomUUID().replace(/-/g, ""),
        "User-Agent":
          "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0",
      },
      body: ssml,
    });

    if (!res.ok) {
      // Fallback: Return structured flag directing client to Web Speech API
      return NextResponse.json({
        fallback_web_speech: true,
        text: cleanText,
        voice,
      });
    }

    const audioData = await res.arrayBuffer();

    return new Response(audioData, {
      headers: {
        "Content-Type": "audio/mpeg",
        "Cache-Control": "public, max-age=86400",
      },
    });
  } catch (err: any) {
    return NextResponse.json(
      { fallback_web_speech: true, error: err.message },
      { status: 200 }
    );
  }
}
