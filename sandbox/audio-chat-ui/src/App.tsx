import { MutableRefObject, useEffect, useRef, useState, useCallback } from "react";
import { Canvas } from "@react-three/fiber";
import { motion, AnimatePresence } from "framer-motion";
import { Mic, Square, Loader2, Info, PhoneOff } from "lucide-react";
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

const DEFAULT_API_BASE = "http://127.0.0.1:8000";
const SILENCE_THRESHOLD = -50; // dB
const SILENCE_DURATION = 3500; // ms (3.5s)

function App() {
  const [apiBase] = useState(DEFAULT_API_BASE);
  const [userId] = useState("demo-user");
  const [conversationId] = useState(crypto.randomUUID());
  
  const [status, setStatus] = useState("Tap to start");
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [showDetails, setShowDetails] = useState(false);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  
  // VAD / Silence detection refs
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const lastActiveTimeRef = useRef<number>(Date.now());

  // Use callback for stopRecording so it can be called reliably from analyzer
  const stopRecording = useCallback(() => {
    if (!mediaRecorderRef.current || mediaRecorderRef.current.state === "inactive") return;
    mediaRecorderRef.current.stop();
    setIsRecording(false);
    setStatus("Processing...");
    
    // Stop the analyzer loop
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
    }
  }, []);

  const monitorVolume = useCallback(() => {
    if (!analyserRef.current) return;
    
    const dataArray = new Float32Array(analyserRef.current.fftSize);
    analyserRef.current.getFloatTimeDomainData(dataArray);
    
    // Calculate RMS volume in dB
    let sumSquares = 0;
    for (let i = 0; i < dataArray.length; i++) {
      sumSquares += dataArray[i] * dataArray[i];
    }
    const rms = Math.sqrt(sumSquares / dataArray.length);
    const db = 20 * Math.log10(rms);

    // If volume is above threshold, reset the silence timer
    if (db > SILENCE_THRESHOLD) {
      lastActiveTimeRef.current = Date.now();
    }

    // Check if we've reached silence duration
    if (Date.now() - lastActiveTimeRef.current > SILENCE_DURATION) {
      console.log("Silence detected, stopping recording...");
      stopRecording();
      return; 
    }

    animationFrameRef.current = requestAnimationFrame(monitorVolume);
  }, [stopRecording]);

  async function startRecording() {
    if (isLoading || isRecording) return;
    setError("");
    setStatus("Listening...");

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      
      // Setup AudioContext for silence detection
      if (!audioContextRef.current) {
        audioContextRef.current = new AudioContext();
      }
      if (audioContextRef.current.state === "suspended") {
        await audioContextRef.current.resume();
      }
      const source = audioContextRef.current.createMediaStreamSource(stream);
      const analyser = audioContextRef.current.createAnalyser();
      analyser.fftSize = 256;
      source.connect(analyser);
      analyserRef.current = analyser;
      
      const recorder = new MediaRecorder(stream);
      chunksRef.current = [];

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };

      recorder.onstop = async () => {
        stream.getTracks().forEach((track) => track.stop());
        const mimeType = recorder.mimeType || "audio/webm";
        const blob = new Blob(chunksRef.current, { type: mimeType });
        await runVoicePipeline(blob, mimeType);
      };

      mediaRecorderRef.current = recorder;
      recorder.start();
      setIsRecording(true);
      
      // Start monitoring
      lastActiveTimeRef.current = Date.now();
      animationFrameRef.current = requestAnimationFrame(monitorVolume);
      
    } catch (err) {
      setError("Microphone unavailable.");
      setStatus("Error");
    }
  }

  async function runVoicePipeline(blob: Blob, mimeType: string) {
    setIsLoading(true);
    try {
      const audioBase64 = await blobToBase64(blob);
      
      setStatus("Interpreting...");
      const stt = await fetchJson<SttResponse>(`${apiBase}/api/stt`, {
        audio_base64: audioBase64,
        audio_mime_type: mimeType,
      });
      const transcript = stt.text.trim();
      if (!transcript) throw new Error("Silence captured.");

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

      setStatus("Speaking...");
      await playResponseAudio(tts.audio_base64, tts.audio_mime_type, audioContextRef);
      setStatus("Listening...");
      setIsLoading(false);
      startRecording();
    } catch (err: any) {
      setError(err.message || "Ready.");
      setStatus("Tap to start");
      setIsLoading(false);
    }
  }

  // End Call function (hard stop)
  const endCall = () => {
    stopRecording();
    setIsLoading(false);
    setStatus("Call ended");
    // Stop any playing audio if needed (could track current audio object)
  };

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
          <button className="icon-btn" onClick={() => setShowDetails(!showDetails)}>
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
              <div className="info-row"><span>Session</span> {conversationId.slice(0,8)}...</div>
              {error && <div className="error-text">{error}</div>}
            </motion.div>
          )}
        </AnimatePresence>

        <footer className="gemini-footer">
          <div className="control-pill-container">
            <motion.button 
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={isRecording || isLoading ? endCall : startRecording}
              className={`mic-button ${(isRecording || isLoading) ? "active danger" : ""}`}
            >
              { (isRecording || isLoading) ? <PhoneOff size={24} /> : <Mic size={24} />}
            </motion.button>
            <div className="tap-hint">
              {isRecording ? "Listening (Auto-Submitting)" : isLoading ? "In Call/Speaking" : "Tap to start call"}
            </div>
          </div>
        </footer>
      </main>
    </div>
  );
}

// Helpers
async function fetchJson<T>(url: string, payload: Record<string, string>) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(await res.text() || `HTTP ${res.status}`);
  return (await res.json()) as T;
}

function blobToBase64(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onloadend = () => {
      const s = r.result as string;
      if (s) resolve(s.split(",")[1]);
      else reject(new Error("Parse fail"));
    };
    r.onerror = () => reject(new Error("Buffer read fault"));
    r.readAsDataURL(blob);
  });
}

function createAudioUrl(base64: string, mime: string) {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return URL.createObjectURL(new Blob([bytes], { type: mime }));
}

function base64ToUint8Array(base64: string) {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i);
  }
  return bytes;
}

async function playResponseAudio(
  base64: string,
  mime: string,
  audioContextRef: MutableRefObject<AudioContext | null>,
) {
  const bytes = base64ToUint8Array(base64);

  if (audioContextRef.current) {
    try {
      if (audioContextRef.current.state === "suspended") {
        await audioContextRef.current.resume();
      }
      const decoded = await audioContextRef.current.decodeAudioData(bytes.buffer.slice(0));
      const source = audioContextRef.current.createBufferSource();
      source.buffer = decoded;
      source.connect(audioContextRef.current.destination);

      await new Promise<void>((resolve) => {
        source.onended = () => resolve();
        source.start(0);
      });
      return;
    } catch (error) {
      console.warn("Web Audio playback failed, falling back to HTMLAudioElement.", error);
    }
  }

  const audioUrl = URL.createObjectURL(new Blob([bytes], { type: mime }));
  try {
    const audio = new Audio(audioUrl);
    await new Promise<void>((resolve, reject) => {
      audio.onended = () => resolve();
      audio.onerror = () => reject(new Error("HTML audio playback failed."));
      void audio.play().catch(reject);
    });
  } finally {
    URL.revokeObjectURL(audioUrl);
  }
}

export default App;
