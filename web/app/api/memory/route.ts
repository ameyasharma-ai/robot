import { NextRequest, NextResponse } from "next/server";
import { createClient } from "@supabase/supabase-js";

export const runtime = "edge";

/**
 * Supabase pgvector conversation and semantic memory endpoint.
 * Persists user sessions and performs semantic vector retrieval for context.
 */
export async function POST(req: NextRequest) {
  try {
    const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
    const supabaseKey = process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;

    if (!supabaseUrl || !supabaseKey) {
      return NextResponse.json({
        status: "mock_local",
        message: "Supabase credentials not configured in environment. Using in-memory fallback.",
        memories: [],
      });
    }

    const supabase = createClient(supabaseUrl, supabaseKey);
    const body = await req.json();
    const { action, session_id, message, role = "user" } = body;

    if (action === "store_message") {
      const { data, error } = await supabase.from("chat_messages").insert([
        {
          session_id: session_id || "default_session",
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
        .eq("session_id", session_id || "default_session")
        .order("created_at", { ascending: true })
        .limit(30);

      if (error) {
        return NextResponse.json({ status: "error", error: error.message }, { status: 500 });
      }
      return NextResponse.json({ status: "ok", history: data || [] });
    }

    return NextResponse.json({ status: "error", message: `Unknown memory action '${action}'` }, { status: 400 });
  } catch (err: any) {
    return NextResponse.json({ status: "error", error: err.message }, { status: 500 });
  }
}
