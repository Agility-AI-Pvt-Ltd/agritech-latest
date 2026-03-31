import { MutableRefObject, useCallback, useEffect, useRef, useState } from "react";
import { MicVAD, utils } from "@ricky0123/vad-web";
import { Canvas } from "@react-three/fiber";
import { motion, AnimatePresence } from "framer-motion";
import { Mic, Loader2, Info, PhoneOff } from "lucide-react";
import AudioOrb from "./components/AudioOrb";

type ChatResponse = {
  response: string;
  conversation_id: string;
  user_id: string;
  tools_used: string[];
  loop_count: number;
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

const DEFAULT_API_BASE = "http://127.0.0.1:8000";
const VAD_ASSET_BASE = new URL("vad/", document.baseURI).toString();
const THREE_SECOND_PAUSE_MS = 3000;
const PRE_SPEECH_PAD_MS = 960;
const MIN_SPEECH_MS = 480;

function App() {
  const [apiBase] = useState(DEFAULT_API_BASE);
  const [userId] = useState("demo-user");
  const [conversationId] = useState(crypto.randomUUID());

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
      pipelineActiveRef.current = true;

      try {
        await vadRef.current?.pause();
      } catch {
        // Ignore pause failures and continue single-flight processing.
      }

      setCapturedSamples(audioFloat32.length);
      setVadEvent("captured");
      setVadStatus(`Got audio - ${audioFloat32.length} samples`);
      console.log("Audio captured:", audioFloat32);

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

        setStatus("Thinking...");
        const chat = await fetchJson<ChatResponse>(`${apiBase}/api/chat`, {
          user_id: userId,
          conversation_id: conversationId,
          query: transcript,
        });

        setStatus("Responding...");
        const tts = await fetchJson<TtsResponse>(`${apiBase}/api/tts`, {
          text: chat.response,
        });

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
    [apiBase, conversationId, resumeListening, setVoiceState, stopPlayback, userId],
  );

  const ensureVAD = useCallback(async () => {
    if (vadRef.current) return vadRef.current;

    const vad = await MicVAD.new({
      baseAssetPath: VAD_ASSET_BASE,
      onnxWASMBasePath: VAD_ASSET_BASE,
      positiveSpeechThreshold: 0.8,
      negativeSpeechThreshold: 0.5,
      minSpeechMs: MIN_SPEECH_MS,
      preSpeechPadMs: PRE_SPEECH_PAD_MS,
      redemptionMs: THREE_SECOND_PAUSE_MS,
      onFrameProcessed: (probabilities) => {
        setSpeechProbability(probabilities.isSpeech);
      },
      onSpeechStart: () => {
        console.log("onSpeechStart");
        setVadEvent("speech");
        setVadStatus("Speaking...");
      },
      onSpeechEnd: async (audioData) => {
        console.log("onSpeechEnd", audioData.length);
        if (!isCallActiveRef.current || pipelineActiveRef.current) return;
        await runVoicePipeline(audioData);
      },
      onVADMisfire: () => {
        console.log("onVADMisfire");
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

    setError("");
    setCapturedSamples(0);
    setSpeechProbability(0);
    setVadEvent("listening");
    setVadStatus("Initializing VAD...");

    try {
      const vad = await ensureVAD();
      isCallActiveRef.current = true;
      await vad.start();
      setVadStatus("Listening...");
      setVoiceState("LISTENING");
    } catch (err) {
      const message = err instanceof Error ? err.message : "Microphone unavailable.";
      setError(message);
      setVadEvent("idle");
      setVadStatus("Idle");
      setVoiceState("IDLE");
      isCallActiveRef.current = false;
    }
  }, [ensureVAD, isLoading, isRecording, setVoiceState]);

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

  return (
    <div className="gemini-shell">
      <div className="visualizer-container">
        <Canvas camera={{ position: [0, 0, 8], fov: 45 }}>
          <ambientLight intensity={0.5} />
          <pointLight position={[10, 10, 10]} intensity={1} />
          <group scale={1.2}>
            <AudioOrb isRecording={isRecording} isThinking={isLoading} />
          </group>
        </Canvas>
      </div>

      <main className="gemini-content">
        <header className="gemini-header">
          <div className="system-status">
            <motion.div
              animate={{ opacity: isRecording || isLoading ? 1 : 0.6 }}
              className="status-pill"
            >
              {isLoading && <Loader2 size={14} className="spin" />}
              <span>{status}</span>
            </motion.div>
          </div>
          <button className="icon-btn" onClick={() => setShowDetails((open) => !open)}>
            <Info size={18} opacity={0.5} />
          </button>
        </header>

        <AnimatePresence>
          {showDetails && (
            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.9 }}
              className="debug-info"
            >
              <div className="info-row"><span>User</span> {userId}</div>
              <div className="info-row"><span>Session</span> {conversationId.slice(0, 8)}...</div>
              <div className="info-row"><span>VAD</span> {vadStatus}</div>
              <div className="info-row"><span>Speech Prob</span> {speechProbability.toFixed(2)}</div>
              <div className="info-row"><span>Samples</span> {capturedSamples || "-"}</div>
              {error && <div className="error-text">{error}</div>}
            </motion.div>
          )}
        </AnimatePresence>

        <footer className="gemini-footer">
          <div className="control-pill-container">
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={isRecording || isLoading ? () => void endCall() : () => void startCall()}
              className={`mic-button ${(isRecording || isLoading) ? "active danger" : ""}`}
            >
              {(isRecording || isLoading) ? <PhoneOff size={24} /> : <Mic size={24} />}
            </motion.button>
            <div className="tap-hint">
              {isRecording ? "Listening with 3s pause detection" : isLoading ? "Processing or speaking" : "Tap to start call"}
            </div>
            <div className={`vad-feedback ${vadEvent}`}>
              {vadStatus}
            </div>
            <div className="tap-hint">Speech probability: {speechProbability.toFixed(2)}</div>
          </div>
        </footer>
      </main>
    </div>
  );
}

async function fetchJson<T>(url: string, payload: Record<string, string>) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error((await res.text()) || `HTTP ${res.status}`);
  return (await res.json()) as T;
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
