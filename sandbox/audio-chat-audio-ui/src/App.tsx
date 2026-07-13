import React, { MutableRefObject, useCallback, useEffect, useRef, useState } from "react";
import { MicVAD, utils } from "@ricky0123/vad-web";
import { Canvas } from "@react-three/fiber";
import { motion, AnimatePresence } from "framer-motion";
import {
  Mic,
  Loader2,
  Info,
  PhoneOff,
  Send,
  ThumbsUp,
  ThumbsDown,
  Copy,
  RefreshCw,
  LogIn,
  LogOut,
  Leaf,
  ShieldCheck,
  CloudSun,
  Database,
  Brain,
} from "lucide-react";
import AudioOrb from "./components/AudioOrb";
import "./styles.css";

type ChatResponse = {
  response: string;
  conversation_id: string;
  user_id: string;
  tools_used: string[];
  loop_count: number;
  rate_limit_remaining: number;
  rate_limit_limit: number;
};

type AuthUser = {
  sub: string;
  email: string;
  name: string;
  picture: string;
};

type ChatQuota = {
  limit: number;
  used: number;
  remaining: number;
};

type AuthResponse = {
  authenticated: boolean;
  user: AuthUser | null;
  chat_quota?: ChatQuota;
};

type SttResponse = {
  text: string;
};

type TtsResponse = {
  audio_base64: string;
  audio_mime_type: string;
};

type VoiceState = "IDLE" | "LISTENING" | "PROCESSING" | "SPEAKING";
type VadEvent = "idle" | "listening" | "speech" | "captured" | "misfire";

type Message = {
  id: string;
  role: "user" | "assistant";
  text: string;
};

const DEFAULT_API_BASE = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";
const VAD_ASSET_BASE = new URL("vad/", document.baseURI).toString();
const THREE_SECOND_PAUSE_MS = 3000;
const PRE_SPEECH_PAD_MS = 960;
const MIN_SPEECH_MS = 480;
const SECURE_CONTEXT_HELP =
  "Voice detection needs HTTPS or localhost. This HTTP EC2 page can run text chat, but browsers block microphone access on public HTTP sites.";

const launchFeatures = [
  {
    title: "Voice-first guidance",
    text: "Speak naturally, get Sarvam-powered transcription, then hear the answer back in Hindi.",
    icon: Mic,
  },
  {
    title: "Grounded crop advice",
    text: "Hybrid retrieval searches maize manuals, POP, fertilizer, pest, disease, and farmer guidance sources.",
    icon: Database,
  },
  {
    title: "Weather-aware context",
    text: "Advisories combine crop stage, location, and forecast risk before suggesting farm operations.",
    icon: CloudSun,
  },
  {
    title: "Safety checked",
    text: "A front-door safety gate keeps the agent focused on safe agricultural advisory queries.",
    icon: ShieldCheck,
  },
  {
    title: "Learns your farm",
    text: "The agent remembers useful profile facts like sowing date, crop stage, location, and language.",
    icon: Brain,
  },
  {
    title: "Maize specialist",
    text: "Built around stage-sensitive maize guidance for fertilizer, irrigation, pest, and disease decisions.",
    icon: Leaf,
  },
];

const pipelineSteps = [
  { label: "Google login", desc: "Secure OAuth — your farm data stays private" },
  { label: "Safety gate", desc: "Front-door filter keeps queries on-topic" },
  { label: "Profile memory", desc: "Remembers sowing date, crop stage & location" },
  { label: "RAG search", desc: "Searches maize manuals, POP & disease guides" },
  { label: "Weather context", desc: "Pulls forecast risk before giving advice" },
  { label: "Kisan Mitra", desc: "Returns answer in text & voice (Hindi)" },
];

function createId() {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
}

function getVoiceSupportError() {
  const isLocalhost = ["localhost", "127.0.0.1", "::1"].includes(window.location.hostname);
  if (!window.isSecureContext && !isLocalhost) {
    return SECURE_CONTEXT_HELP;
  }
  if (!navigator.mediaDevices?.getUserMedia) {
    return "Microphone API is unavailable in this browser context. Use HTTPS, localhost, or a browser with microphone support enabled.";
  }
  return "";
}

function getFriendlyVoiceError(err: unknown) {
  if (err instanceof DOMException) {
    if (err.name === "NotAllowedError") {
      return "Microphone permission was blocked. Allow microphone access in the browser and try again.";
    }
    if (err.name === "NotFoundError") {
      return "No microphone was found on this device.";
    }
  }

  const message = err instanceof Error ? err.message : "Microphone unavailable.";
  if (message.includes("mediaDevices") || message.includes("getUserMedia")) {
    return SECURE_CONTEXT_HELP;
  }
  return message;
}

function App() {
  const [apiBase] = useState(DEFAULT_API_BASE);
  const [conversationId] = useState(createId());
  const [authUser, setAuthUser] = useState<AuthUser | null>(null);
  const [authLoading, setAuthLoading] = useState(true);
  const [authError, setAuthError] = useState("");
  const [chatQuota, setChatQuota] = useState<ChatQuota | null>(null);
  const userId = authUser?.sub ?? "";

  // Chat State
  const [messages, setMessages] = useState<Message[]>([
    { id: "1", role: "assistant", text: "नमस्ते! मैं किसान मित्र हूँ। आज मैं आपकी खेती की ज़रूरतों में कैसे मदद कर सकता हूँ?" }
  ]);
  const [inputText, setInputText] = useState("");
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Voice State
  const [status, setStatus] = useState("Tap to start");
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [showDetails, setShowDetails] = useState(false);
  const [vadStatus, setVadStatus] = useState("Idle");
  const [vadEvent, setVadEvent] = useState<VadEvent>("idle");
  const [speechProbability, setSpeechProbability] = useState(0);
  const [capturedSamples, setCapturedSamples] = useState(0);

  const vadRef = useRef<MicVAD | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const isCallActiveRef = useRef(false);
  const mountedRef = useRef(true);
  const currentAudioRef = useRef<HTMLAudioElement | null>(null);
  const currentSourceRef = useRef<AudioBufferSourceNode | null>(null);
  const pipelineActiveRef = useRef(false);

  const refreshAuth = useCallback(async () => {
    setAuthLoading(true);
    setAuthError("");
    try {
      const auth = await getJson<AuthResponse>(`${apiBase}/auth/me`);
      setAuthUser(auth.authenticated ? auth.user : null);
      setChatQuota(auth.chat_quota ?? null);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Could not check Google login.";
      setAuthError(message);
      setAuthUser(null);
      setChatQuota(null);
    } finally {
      setAuthLoading(false);
    }
  }, [apiBase]);

  useEffect(() => {
    void refreshAuth();
  }, [refreshAuth]);

  const handleGoogleLogin = () => {
    window.location.assign(`${apiBase}/auth/google/login`);
  };

  const handleLogout = async () => {
    await postJson(`${apiBase}/auth/logout`, {});
    isCallActiveRef.current = false;
    pipelineActiveRef.current = false;
    currentAudioRef.current?.pause();
    await vadRef.current?.pause();
    setAuthUser(null);
    setChatQuota(null);
    setIsRecording(false);
    setIsLoading(false);
  };

  // Scroll to bottom when messages get added
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };
  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const setVoiceState = useCallback((nextState: VoiceState) => {
    if (!mountedRef.current) return;

    setIsRecording(nextState === "LISTENING");
    setIsLoading(nextState === "PROCESSING" || nextState === "SPEAKING");

    if (nextState === "IDLE") setStatus("Tap to start");
    if (nextState === "LISTENING") setStatus("Listening...");
    if (nextState === "PROCESSING") setStatus("Processing...");
    if (nextState === "SPEAKING") setStatus("Speaking...");
  }, []);

  const stopPlayback = useCallback(() => {
    currentAudioRef.current?.pause();
    currentAudioRef.current = null;

    if (currentSourceRef.current) {
      currentSourceRef.current.onended = null;
      try {
        currentSourceRef.current.stop();
      } catch {
        // Ignore already-stopped sources.
      }
      currentSourceRef.current.disconnect();
      currentSourceRef.current = null;
    }
  }, []);

  const resumeListening = useCallback(async () => {
    if (!isCallActiveRef.current || !vadRef.current) return;
    pipelineActiveRef.current = false;
    await vadRef.current.start();
    setVadEvent("listening");
    setVadStatus("Listening...");
    setVoiceState("LISTENING");
  }, [setVoiceState]);

  const runVoicePipeline = useCallback(
    async (audioFloat32: Float32Array) => {
      if (!isCallActiveRef.current || pipelineActiveRef.current) return;
      if (!authUser || (chatQuota && chatQuota.remaining <= 0)) {
        setError("Please sign in with Google and make sure you have chats remaining.");
        return;
      }
      pipelineActiveRef.current = true;

      try {
        await vadRef.current?.pause();
      } catch {
      }

      setCapturedSamples(audioFloat32.length);
      setVadEvent("captured");
      setVadStatus(`Got audio - ${audioFloat32.length} samples`);

      setVoiceState("PROCESSING");
      setStatus("Interpreting...");
      setError("");

      try {
        const wavBlob = float32ToWavBlob(audioFloat32);
        const audioBase64 = await blobToBase64(wavBlob);

        const stt = await fetchJson<SttResponse>(`${apiBase}/api/stt`, {
          audio_base64: audioBase64,
          audio_mime_type: wavBlob.type,
        });

        const transcript = stt.text.trim();
        if (!transcript) {
          throw new Error("Speech captured but transcript was empty.");
        }

        // Add user Speech to Chat
        setMessages(prev => [...prev, { id: createId(), role: "user", text: transcript }]);

        setStatus("Thinking...");
        const chat = await fetchJson<ChatResponse>(`${apiBase}/api/chat`, {
          user_id: userId,
          conversation_id: conversationId,
          query: transcript,
        });
        setChatQuota({
          limit: chat.rate_limit_limit,
          used: chat.rate_limit_limit - chat.rate_limit_remaining,
          remaining: chat.rate_limit_remaining,
        });

        setStatus("Responding...");
        // Fetch TTS
        const tts = await fetchJson<TtsResponse>(`${apiBase}/api/tts`, {
          text: chat.response,
        });

        // Add AI response to Chat
        setMessages(prev => [...prev, { id: createId(), role: "assistant", text: chat.response }]);

        if (!isCallActiveRef.current) return;

        setVoiceState("SPEAKING");
        setVadStatus("Speaking response...");
        stopPlayback();
        await playResponseAudio(
          tts.audio_base64,
          tts.audio_mime_type,
          audioContextRef,
          currentAudioRef,
          currentSourceRef,
        );

        await resumeListening();
      } catch (err) {
        const message = err instanceof Error ? err.message : "Voice pipeline failed.";
        setError(message);
        pipelineActiveRef.current = false;
        setVoiceState("IDLE");
        setVadEvent("idle");
        setVadStatus("Idle");
        isCallActiveRef.current = false;
      }
    },
    [apiBase, authUser, chatQuota, conversationId, resumeListening, setVoiceState, stopPlayback, userId],
  );

  const handleSendText = async () => {
    if (!inputText.trim() || isLoading) return;
    if (!authUser) {
      setError("Please sign in with Google to use chat.");
      return;
    }
    if (chatQuota && chatQuota.remaining <= 0) {
      setError("Chat limit reached for this session.");
      return;
    }
    const query = inputText.trim();
    setInputText("");
    
    // Optimistic UI update
    setMessages(prev => [...prev, { id: createId(), role: "user", text: query }]);
    setIsLoading(true);

    try {
      const chat = await fetchJson<ChatResponse>(`${apiBase}/api/chat`, {
        user_id: userId,
        conversation_id: conversationId,
        query: query,
      });
      setChatQuota({
        limit: chat.rate_limit_limit,
        used: chat.rate_limit_limit - chat.rate_limit_remaining,
        remaining: chat.rate_limit_remaining,
      });

      setMessages(prev => [...prev, { id: createId(), role: "assistant", text: chat.response }]);
      
      // If call is active, it speaks out the text chat response too
      if (isCallActiveRef.current) {
        setVoiceState("PROCESSING");
        pipelineActiveRef.current = true;
        try {
          await vadRef.current?.pause();
        } catch { }
        
        const tts = await fetchJson<TtsResponse>(`${apiBase}/api/tts`, {
            text: chat.response,
        });
        
        setVoiceState("SPEAKING");
        setVadStatus("Speaking response...");
        stopPlayback();
        await playResponseAudio(
          tts.audio_base64,
          tts.audio_mime_type,
          audioContextRef,
          currentAudioRef,
          currentSourceRef,
        );
        resumeListening();
      }
    } catch (err) {
      console.error(err);
      setError(err instanceof Error ? err.message : "Failed to process text message.");
    } finally {
      if (!isCallActiveRef.current) setIsLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSendText();
    }
  }

  const ensureVAD = useCallback(async () => {
    if (vadRef.current) return vadRef.current;

    const vad = await MicVAD.new({
      baseAssetPath: VAD_ASSET_BASE,
      onnxWASMBasePath: VAD_ASSET_BASE,
      processorType: "ScriptProcessor",
      ortConfig: (ort) => {
        ort.env.logLevel = "error";
        ort.env.wasm.numThreads = 1;
      },
      positiveSpeechThreshold: 0.8,
      negativeSpeechThreshold: 0.5,
      minSpeechMs: MIN_SPEECH_MS,
      preSpeechPadMs: PRE_SPEECH_PAD_MS,
      redemptionMs: THREE_SECOND_PAUSE_MS,
      onFrameProcessed: (probabilities) => {
        setSpeechProbability(probabilities.isSpeech);
      },
      onSpeechStart: () => {
        setVadEvent("speech");
        setVadStatus("Speaking...");
      },
      onSpeechEnd: async (audioData) => {
        if (!isCallActiveRef.current || pipelineActiveRef.current) return;
        await runVoicePipeline(audioData);
      },
      onVADMisfire: () => {
        setVadEvent("misfire");
        setVadStatus("Misfire (noise, not speech)");
      },
      startOnLoad: false,
    });

    vadRef.current = vad;
    return vad;
  }, [runVoicePipeline]);

  const startCall = useCallback(async () => {
    if (isLoading || isRecording) return;
    if (!authUser) {
      setError("Please sign in with Google to use voice chat.");
      return;
    }
    if (chatQuota && chatQuota.remaining <= 0) {
      setError("Chat limit reached for this session.");
      return;
    }

    setError("");
    setCapturedSamples(0);
    setSpeechProbability(0);
    setVadEvent("listening");
    setVadStatus("Initializing VAD...");

    try {
      const voiceSupportError = getVoiceSupportError();
      if (voiceSupportError) {
        throw new Error(voiceSupportError);
      }

      const vad = await ensureVAD();
      isCallActiveRef.current = true;
      await vad.start();
      setVadStatus("Listening...");
      setVoiceState("LISTENING");
    } catch (err) {
      setError(getFriendlyVoiceError(err));
      setVadEvent("idle");
      setVadStatus("Idle");
      setVoiceState("IDLE");
      isCallActiveRef.current = false;
    }
  }, [authUser, chatQuota, ensureVAD, isLoading, isRecording, setVoiceState]);

  const endCall = useCallback(async () => {
    isCallActiveRef.current = false;
    pipelineActiveRef.current = false;
    stopPlayback();

    try {
      await vadRef.current?.pause();
    } catch {
      // Ignore pause failures on teardown.
    }

    setSpeechProbability(0);
    setCapturedSamples(0);
    setVadEvent("idle");
    setVadStatus("Idle");
    setVoiceState("IDLE");
  }, [setVoiceState, stopPlayback]);

  useEffect(() => {
    return () => {
      mountedRef.current = false;
      isCallActiveRef.current = false;
      pipelineActiveRef.current = false;
      stopPlayback();
      void vadRef.current?.destroy();
      void audioContextRef.current?.close();
    };
  }, [stopPlayback]);

  if (authLoading) {
    return (
      <div className="auth-screen">
        <div className="auth-panel">
          <Loader2 className="spin" size={28} />
          <p>Checking Google session...</p>
        </div>
      </div>
    );
  }

  if (!authUser) {
    return (
      <div className="launch-page">
        {/* Ambient video (very subtle opacity) */}
        <video className="launch-video-bg" autoPlay muted loop playsInline poster="/bg.png" aria-hidden="true">
          <source src="/home-background.mp4" type="video/mp4" />
        </video>

        {/* ── STICKY NAV ── */}
        <nav className="launch-nav" aria-label="Main navigation">
          <div className="auth-brand">
            <svg width="28" height="28" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" className="logo-icon">
              <path d="M12 2L2 7L12 12L22 7L12 2Z" fill="#D96941"/>
              <path d="M2 17L12 22L22 17" stroke="#3DBFB7" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              <path d="M2 12L12 17L22 12" stroke="#D96941" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
            <span>Kisan Mitra</span>
          </div>
          <button className="launch-login compact" onClick={handleGoogleLogin}>
            <LogIn size={16} />
            Sign in
          </button>
        </nav>

        {/* ── HERO ── */}
        <main className="launch-hero-section">
          <div className="launch-copy">
            <span className="launch-kicker">
              <span className="launch-kicker-dot" />
              Agricultural AI for Indian Farmers
            </span>
            <h1>Kisan<br />Mitra</h1>
            <p>
              A voice-first crop assistant that listens in Hindi, searches trusted
              maize knowledge, checks weather risk, and delivers practical farm advice.
            </p>
            {authError && <div className="auth-error launch-error">{authError}</div>}
            <div className="launch-cta-group">
              <button className="launch-login" onClick={handleGoogleLogin}>
                <LogIn size={18} />
                Continue with Google
              </button>
              <a className="launch-learn-more" href="#features">
                Learn more ↓
              </a>
            </div>
          </div>

          <div className="agent-preview" aria-label="Kisan Mitra agent preview">
            <div className="preview-topline">
              <span>Live conversation</span>
              <span className="preview-status">
                <span className="preview-status-dot" />
                Ready
              </span>
            </div>
            <div className="preview-message farmer">
              🌽 My maize leaves have yellow spots. What should I do today?
            </div>
            <div className="preview-message agent">
              Checking crop stage, disease database and today's weather forecast…
            </div>
            <div className="preview-typing">
              <span className="typing-dot" />
              <span className="typing-dot" />
              <span className="typing-dot" />
            </div>
            <div className="preview-meter">
              <span /><span /><span /><span />
            </div>
          </div>
        </main>

        {/* ── FEATURES ── */}
        <section id="features" className="launch-features-section" aria-label="Kisan Mitra features">
          <p className="section-eyebrow">Capabilities</p>
          <h2 className="section-heading">Everything a farmer needs</h2>
          <p className="section-subheading">
            Six specialised capabilities working together to give you trusted, safe, and localised crop guidance.
          </p>
          <div className="launch-features">
            {launchFeatures.map((feature) => {
              const Icon = feature.icon;
              return (
                <article className="feature-card" key={feature.title}>
                  <div className="feature-icon">
                    <Icon size={20} />
                  </div>
                  <h2>{feature.title}</h2>
                  <p>{feature.text}</p>
                </article>
              );
            })}
          </div>
        </section>

        {/* ── HOW IT WORKS ── */}
        <section className="pipeline-section" aria-label="How Kisan Mitra works">
          <p className="section-eyebrow">How it works</p>
          <h2 className="section-heading">Six-step intelligent pipeline</h2>
          <p className="section-subheading">
            Every query passes through a structured pipeline designed for safety, accuracy, and context-awareness.
          </p>
          <div className="pipeline-steps-row">
            {pipelineSteps.map((step, index) => (
              <div className="pipeline-step" key={step.label}>
                <div className="pipeline-step-num">{String(index + 1).padStart(2, "0")}</div>
                <div className="pipeline-step-label">{step.label}</div>
                <div className="pipeline-step-desc">{step.desc}</div>
              </div>
            ))}
          </div>
        </section>

        {/* ── BOTTOM CTA ── */}
        <section className="launch-cta-section" aria-label="Call to action">
          <h2>Ready to talk to your farm?</h2>
          <p>Sign in with Google in seconds. No app download required — works on any device, in Hindi.</p>
          <button className="launch-login" onClick={handleGoogleLogin}>
            <LogIn size={18} />
            Get started — it's free
          </button>
        </section>

        {/* ── SITE FOOTER ── */}
        <footer className="launch-site-footer">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M12 2L2 7L12 12L22 7L12 2Z" fill="#D96941"/>
            <path d="M2 17L12 22L22 17" stroke="#3DBFB7" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
          <span>Kisan Mitra · Agility AI · Built for Indian farmers</span>
        </footer>
      </div>
    );
  }

  return (
    <div className="app-wrapper">
      <div className="main-container">
        
        {/* --- LEFT PANE (Chat) --- */}
        <div className="left-pane">
          <header className="brand-header">
            <div className="brand-mark">
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" className="logo-icon">
                <path d="M12 2L2 7L12 12L22 7L12 2Z" fill="#DC633A"/>
                <path d="M2 17L12 22L22 17" stroke="#3EBFB8" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                <path d="M2 12L12 17L22 12" stroke="#DC633A" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
              <span className="logo-text">Kisan Mitra</span>
            </div>
            <div className="account-pill">
              {chatQuota && (
                <span className={`quota-chip ${chatQuota.remaining <= 0 ? "empty" : ""}`}>
                  {chatQuota.remaining}/{chatQuota.limit} chats
                </span>
              )}
              <span className="account-name">{authUser.name || authUser.email}</span>
              <button className="icon-btn logout-btn" onClick={() => void handleLogout()} title="Sign out">
                <LogOut size={16} />
              </button>
            </div>
          </header>

          <div className="chat-messages">
            {messages.map((msg) => (
              <div key={msg.id} className={`message ${msg.role}`}>
                <div className="message-header">
                  {msg.role === "assistant" ? (
                    <div className="avatar ai">
                      <div className="orb-icon" />
                    </div>
                  ) : (
                    <div className="avatar">
                      {authUser.picture ? <img src={authUser.picture} alt="User" /> : "U"}
                    </div>
                  )}
                  <span className="sender-name">{msg.role === "user" ? "You" : "Kisan Mitra"}</span>
                </div>
                <div className="message-bubble">
                  {msg.text}
                </div>
                {msg.role === "assistant" && (
                  <div className="message-actions">
                    <div className="action-icons">
                      <button className="action-btn"><ThumbsUp size={16} /></button>
                      <button className="action-btn"><ThumbsDown size={16} /></button>
                      <button className="action-btn"><Copy size={16} /></button>
                      <button className="action-btn"><RefreshCw size={16} /></button>
                    </div>
                  </div>
                )}
              </div>
            ))}
            {isLoading && !isCallActiveRef.current && (
              <div className="message assistant">
                 <div className="message-header">
                  <div className="avatar ai"><div className="orb-icon" /></div>
                  <span className="sender-name">Kisan Mitra</span>
                </div>
                <div className="message-bubble">
                  <Loader2 size={16} className="spin" />
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          <div className="chat-input-wrapper">
            <div className="chat-input-container">
              <textarea
                className="chat-input"
                placeholder={chatQuota && chatQuota.remaining <= 0 ? "Chat limit reached" : "मैं आपकी कैसे मदद करूँ?"}
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                onKeyDown={handleKeyDown}
                disabled={(isLoading && !isCallActiveRef.current) || Boolean(chatQuota && chatQuota.remaining <= 0)}
              />
              <button 
                className="send-btn" 
                onClick={handleSendText}
                disabled={!inputText.trim() || (isLoading && !isCallActiveRef.current) || Boolean(chatQuota && chatQuota.remaining <= 0)}
              >
                <Send size={18} />
              </button>
            </div>
          </div>
        </div>

        {/* --- RIGHT PANE (Audio Interface) --- */}
        <div className="right-pane">
          <header className="nav-header">
            <button className="icon-btn info-toggle" onClick={() => setShowDetails((open) => !open)}>
              <Info size={18} opacity={0.6} />
            </button>
          </header>

          {/* 3D Audio Visualizer inside the right pane */}
          <div className="visualizer-container">
            <Canvas camera={{ position: [0, 0, 8], fov: 45 }}>
              <ambientLight intensity={0.5} />
              <pointLight position={[10, 10, 10]} intensity={1} />
              <group scale={1.2}>
                <AudioOrb isRecording={isRecording} isThinking={isLoading} />
              </group>
            </Canvas>
          </div>

          <AnimatePresence>
            {showDetails && (
              <motion.div
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.9 }}
                className="debug-info"
              >
                <div className="info-row"><span>User</span> {authUser.email || userId}</div>
                {chatQuota && <div className="info-row"><span>Chats</span> {chatQuota.remaining}/{chatQuota.limit}</div>}
                <div className="info-row"><span>Session</span> {conversationId.slice(0, 8)}...</div>
                <div className="info-row"><span>VAD</span> {vadStatus}</div>
                <div className="info-row"><span>Speech Prob</span> {speechProbability.toFixed(2)}</div>
                <div className="info-row"><span>Samples</span> {capturedSamples || "-"}</div>
                {error && <div className="error-text">{error}</div>}
              </motion.div>
            )}
          </AnimatePresence>

          <footer className="gemini-footer">
            <div className="glass-control-panel">
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={isRecording || isLoading ? () => void endCall() : () => void startCall()}
                className={`mic-button ${(isRecording || isLoading) ? "active danger" : ""}`}
                disabled={Boolean(!isRecording && !isLoading && chatQuota && chatQuota.remaining <= 0)}
              >
                {(isRecording || isLoading) ? <PhoneOff size={24} /> : <Mic size={24} />}
              </motion.button>
              
              <div className="vad-info-stack">
                <div className={`vad-feedback ${vadEvent}`}>
                  {vadStatus}
                </div>
                <div className="tap-hint">
                  {isRecording ? "Listening with 3s pause detection" : isLoading ? "Processing" : "Tap to start voice"}
                </div>
                <div className="speech-prob-hint">
                  Probability: {speechProbability.toFixed(2)}
                </div>
              </div>
            </div>
          </footer>
        </div>

      </div>
    </div>
  );
}

// Below are the underlying network and audio API helpers

async function getJson<T>(url: string) {
  const res = await fetch(url, {
    method: "GET",
    credentials: "include",
  });
  if (!res.ok) throw new Error(await readError(res));
  return (await res.json()) as T;
}

async function postJson<T = unknown>(url: string, payload: Record<string, unknown>) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    credentials: "include",
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(await readError(res));
  return (await res.json()) as T;
}

async function fetchJson<T>(url: string, payload: Record<string, unknown>) {
  return postJson<T>(url, payload);
}

async function readError(res: Response) {
  const fallback = `HTTP ${res.status}`;
  const raw = await res.text();
  if (!raw) return fallback;
  try {
    const data = JSON.parse(raw);
    if (typeof data?.detail === "string") return data.detail;
    return JSON.stringify(data);
  } catch {
    return raw;
  }
}

function blobToBase64(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => {
      const result = reader.result as string;
      if (result) resolve(result.split(",")[1] ?? "");
      else reject(new Error("Parse fail"));
    };
    reader.onerror = () => reject(new Error("Buffer read fault"));
    reader.readAsDataURL(blob);
  });
}

function float32ToWavBlob(audio: Float32Array) {
  const wav = utils.encodeWAV(audio);
  return new Blob([wav], { type: "audio/wav" });
}

function base64ToUint8Array(base64: string) {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) {
    bytes[i] = binary.charCodeAt(i);
  }
  return bytes;
}

async function playResponseAudio(
  base64: string,
  mime: string,
  audioContextRef: MutableRefObject<AudioContext | null>,
  currentAudioRef: MutableRefObject<HTMLAudioElement | null>,
  currentSourceRef: MutableRefObject<AudioBufferSourceNode | null>,
) {
  const bytes = base64ToUint8Array(base64);

  if (!audioContextRef.current) {
    audioContextRef.current = new AudioContext();
  }

  try {
    if (audioContextRef.current.state === "suspended") {
      await audioContextRef.current.resume();
    }

    const decoded = await audioContextRef.current.decodeAudioData(bytes.buffer.slice(0));
    const source = audioContextRef.current.createBufferSource();
    source.buffer = decoded;
    source.connect(audioContextRef.current.destination);
    currentSourceRef.current = source;

    await new Promise<void>((resolve) => {
      source.onended = () => {
        currentSourceRef.current = null;
        resolve();
      };
      source.start(0);
    });
    return;
  } catch (error) {
    console.warn("Web Audio playback failed, falling back to HTMLAudioElement.", error);
  }

  const audioUrl = URL.createObjectURL(new Blob([bytes], { type: mime }));
  try {
    const audio = new Audio(audioUrl);
    currentAudioRef.current = audio;
    await new Promise<void>((resolve, reject) => {
      audio.onended = () => {
        currentAudioRef.current = null;
        resolve();
      };
      audio.onerror = () => {
        currentAudioRef.current = null;
        reject(new Error("HTML audio playback failed."));
      };
      void audio.play().catch(reject);
    });
  } finally {
    URL.revokeObjectURL(audioUrl);
  }
}

export default App;
