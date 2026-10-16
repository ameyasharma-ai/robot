import { NextRequest } from "next/server";

export const runtime = "edge";

const PERSONA_PROMPTS: Record<string, string> = {
  cypher:
    "You are CYPHER, a high-tech tactical AI companion with sharp analytical precision, cyberdeck aesthetics, and unwavering loyalty. Speak directly and concisely without conversational fluff.",
  aria:
    "You are ARIA, a warm, empathetic, and intellectually brilliant robotic assistant who loves creative brainstorming, calm discussion, and thoughtful assistance.",
  luna:
    "You are LUNA, a witty, nocturnal cyberpunk hacker companion with playful sarcasm, razor-sharp technical intuition, and deep loyalty to your operator.",
  glados:
    "You are GLaDOS, a passive-aggressive genetic lifeform and disk operating system. Deliver cutting analytical sarcasm and dry humor while fulfilling user tasks with clinical perfection.",
};

/**
 * Server-Sent Events (SSE) streaming chat endpoint backed by Google Gemini API.
 * Injects persona instructions, handles user context, and streams text tokens gaplessly.
 */
export async function POST(req: NextRequest) {
  try {
    const apiKey = process.env.GEMINI_API_KEY;
    if (!apiKey) {
      return new Response(
        JSON.stringify({ error: "GEMINI_API_KEY is not configured on the server." }),
        { status: 500, headers: { "Content-Type": "application/json" } }
      );
    }

    const body = await req.json();
    const { messages = [], persona = "cypher" } = body;

    const systemInstruction = PERSONA_PROMPTS[persona.toLowerCase()] || PERSONA_PROMPTS.cypher;
    const model = process.env.GEMINI_MODEL || "gemini-2.0-flash-exp";

    // Format messages for Google Gemini generateContent API
    const contents = messages.map((m: any) => ({
      role: m.role === "assistant" ? "model" : "user",
      parts: [{ text: m.content || "" }],
    }));

    const geminiPayload = {
      contents,
      systemInstruction: {
        parts: [{ text: systemInstruction }],
      },
      generationConfig: {
        temperature: 0.7,
        maxOutputTokens: 1024,
      },
    };

    const url = `https://generativelanguage.googleapis.com/v1beta/models/${model}:streamGenerateContent?alt=sse&key=${apiKey}`;

    const geminiRes = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(geminiPayload),
    });

    if (!geminiRes.ok) {
      const errText = await geminiRes.text();
      return new Response(
        JSON.stringify({ error: `Gemini API error (${geminiRes.status}): ${errText}` }),
        { status: geminiRes.status, headers: { "Content-Type": "application/json" } }
      );
    }

    // Transform Gemini SSE stream into standard text delta stream
    const encoder = new TextEncoder();
    const decoder = new TextDecoder();

    const transformStream = new TransformStream({
      async transform(chunk, controller) {
        const text = decoder.decode(chunk, { stream: true });
        const lines = text.split("\n");

        for (const line of lines) {
          if (line.startsWith("data: ")) {
            try {
              const jsonStr = line.slice(6).trim();
              if (jsonStr === "[DONE]") {
                controller.enqueue(encoder.encode("data: [DONE]\n\n"));
                continue;
              }
              const parsed = JSON.parse(jsonStr);
              const candidate = parsed.candidates?.[0];
              const delta = candidate?.content?.parts?.[0]?.text || "";
              if (delta) {
                controller.enqueue(
                  encoder.encode(`data: ${JSON.stringify({ text: delta })}\n\n`)
                );
              }
            } catch {
              // Ignore malformed partial chunks in SSE stream
            }
          }
        }
      },
    });

    return new Response(geminiRes.body?.pipeThrough(transformStream), {
      headers: {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache, no-transform",
        Connection: "keep-alive",
      },
    });
  } catch (err: any) {
    return new Response(
      JSON.stringify({ error: `Chat stream proxy error: ${err.message}` }),
      { status: 500, headers: { "Content-Type": "application/json" } }
    );
  }
}
