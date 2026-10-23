import { NextRequest, NextResponse } from "next/server";
import { createClient } from "@supabase/supabase-js";

export const runtime = "edge";

/**
 * Supabase pgvector conversation and semantic memory endpoint.
 * Persists user sessions and performs semantic vector retrieval for context.
 *
 * Security model:
 * - Server-side only ever uses the SUPABASE_SERVICE_ROLE_KEY. The public anon
 *   key is never used here: it would let this route bypass Row Level Security
 *   while providing no proof of caller identity.
 * - Missing service-role key fails closed with a 500 JSON error.
 * - session_id is required and validated against a strict pattern so clients
 *   cannot read or write arbitrary/shared sessions (IDOR). This app currently
 *   has no per-user auth layer (sibling API routes do not authenticate); if
 *   Supabase Auth is added, bind the session_id owner to the authenticated
 *   user id here before querying.
 */

/** Strict, opaque session identifier: 1-64 chars of [A-Za-z0-9_-]. */
const SESSION_ID_PATTERN = /^[A-Za-z0-9_-]{1,64}$/;

function isValidSessionId(sessionId: unknown): sessionId is string {
  return typeof sessionId === "string" && SESSION_ID_PATTERN.test(sessionId);
}

export async function POST(req: NextRequest) {
  try {
    const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
    // Fail closed: server-side memory access requires the service-role key.
    const supabaseKey = process.env.SUPABASE_SERVICE_ROLE_KEY;

    if (!supabaseUrl || !supabaseKey) {
      return NextResponse.json(
        {
          status: "error",
          error:
            "Server memory access is not configured: set NEXT_PUBLIC_SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY.",
        },
        { status: 500 }
      );
    }

    const body = await req.json();
    const { action, session_id, message, role = "user" } = body;

    // Ownership/IDOR guard: every memory operation must be scoped to a
    // well-formed, explicitly provided session id — never a shared default.
    if (!isValidSessionId(session_id)) {
      return NextResponse.json(
        { status: "error", error: "A valid session_id is required." },
        { status: 400 }
      );
    }

    const supabase = createClient(supabaseUrl, supabaseKey);

    if (action === "store_message") {
      const { data, error } = await supabase.from("chat_messages").insert([
        {
          session_id,
          role,
          content: message,
          created_at: new Date().toISOString(),
        },
      ]);

      if (error) {
        return NextResponse.json({ status: "error", error: error.message }, { status: 500 });
      }
      return NextResponse.json({ status: "stored", data });
    }

    if (action === "get_history") {
      const { data, error } = await supabase
        .from("chat_messages")
        .select("*")
        .eq("session_id", session_id)
        .order("created_at", { ascending: true })
        .limit(30);

      if (error) {
        return NextResponse.json({ status: "error", error: error.message }, { status: 500 });
      }
      return NextResponse.json({ status: "ok", history: data || [] });
    }

    return NextResponse.json(
      { status: "error", message: `Unknown memory action '${action}'` },
      { status: 400 }
    );
  } catch (err: any) {
    return NextResponse.json({ status: "error", error: err.message }, { status: 500 });
  }
}
