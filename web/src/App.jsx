import { useEffect, useState } from 'react'
import { useAgentSocket } from './useAgentSocket.js'
import CameraMesh from './CameraMesh.jsx'

function Dot({ color, pulse }) {
  return (
    <span
      className="inline-block w-1.5 h-1.5 rounded-full shrink-0"
      style={{ background: color, boxShadow: `0 0 8px ${color}`, animation: pulse ? 'pulseDot 1.4s ease-in-out infinite' : 'none' }}
    />
  )
}

const STATUS = {
  listening: { c: '#34d399', l: 'Live', pulse: true },
  'demo running': { c: '#34d399', l: 'Live', pulse: true },
  thinking: { c: '#818cf8', l: 'Thinking', pulse: true },
  idle: { c: '#71717a', l: 'Idle', pulse: false },
  connecting: { c: '#71717a', l: 'Connecting', pulse: true },
  disconnected: { c: '#fb7185', l: 'Offline', pulse: false },
}

function Chip({ children, color }) {
  return (
    <span className="text-[11px] px-2.5 py-1 rounded-md border border-white/10 flex items-center gap-1.5 text-zinc-400 whitespace-nowrap">
      {color && <Dot color={color} />}
      {children}
    </span>
  )
}

function Ctl({ children, onClick, title }) {
  return (
    <button
      onClick={onClick}
      title={title}
      className="text-sm text-zinc-300 px-3 py-1.5 rounded-lg border border-white/10 bg-white/[0.03] hover:border-white/25 hover:text-white transition"
    >
      {children}
    </button>
  )
}

function Waveform({ level }) {
  const N = 36
  const [h, setH] = useState(() => Array(N).fill(0.06))
  useEffect(() => {
    const active = level > 0.03
    setH(Array.from({ length: N }, () => (active ? Math.min(1, 0.12 + Math.random() * Math.min(1, level * 5)) : 0.06)))
  }, [level])
  return (
    <div className="flex items-center justify-center gap-[3px] h-12 mb-4">
      {h.map((v, i) => (
        <div
          key={i}
          className="w-1 rounded-full"
          style={{ height: '100%', transformOrigin: 'center', transform: `scaleY(${v})`, transition: 'transform .12s ease', background: 'linear-gradient(180deg,#a1a1aa,#3f3f46)' }}
        />
      ))}
    </div>
  )
}

function PanelHeader({ dot, title, sub }) {
  return (
    <div className="mb-4">
      <div className="flex items-center gap-2">
        <Dot color={dot} />
        <span className="text-[13px] font-semibold tracking-tight text-zinc-200">{title}</span>
      </div>
      <div className="mono text-[11px] text-zinc-600 mt-1 pl-3.5">{sub}</div>
    </div>
  )
}

function Idle() {
  return (
    <div className="flex flex-col items-center gap-3">
      <span className="w-3 h-3 rounded-full" style={{ background: '#52525b', animation: 'breathe 2.4s ease-in-out infinite' }} />
      <span className="text-[13px] text-zinc-500">listening for a plan…</span>
    </div>
  )
}

function Thinking() {
  return (
    <div className="flex flex-col items-center gap-4">
      <div className="flex items-center gap-1.5">
        {[0, 0.15, 0.3].map((d, i) => (
          <span key={i} className="w-2 h-2 rounded-full bg-indigo-400" style={{ animation: `think 1s ease-in-out ${d}s infinite` }} />
        ))}
      </div>
      <span className="text-[13px] text-zinc-400">reading intent…</span>
    </div>
  )
}

function RouterTrace({ lines }) {
  return (
    <div className="rounded-xl border border-indigo-400/20 bg-indigo-500/[0.04] px-4 py-3" style={{ animation: 'popIn .35s ease both' }}>
      <div className="flex items-center gap-2 mb-2">
        <span className="w-1.5 h-1.5 rounded-full bg-indigo-400" style={{ animation: 'pulseDot 1s infinite' }} />
        <span className="mono text-[10px] tracking-wide text-indigo-300/80">AGENT · REASONING</span>
      </div>
      <div className="flex flex-col gap-1 mono text-[11px] leading-relaxed">
        {lines.map((l, i) => {
          const url = (l.match(/https?:\/\/\S+/) || [])[0]
          return (
            <div key={i} className="flex gap-1.5" style={{ animation: 'fadeUp .25s ease both' }}>
              <span className="text-zinc-600 shrink-0">›</span>
              <span className="text-zinc-400 break-all">
                {url
                  ? <>{l.split(url)[0]}<a href={url} target="_blank" rel="noreferrer" className="text-indigo-300 underline">{url}</a></>
                  : l}
              </span>
            </div>
          )
        })}
      </div>
    </div>
  )
}

function ActionCard({ c }) {
  return (
    <div
      className="relative rounded-xl border bg-zinc-900/60 backdrop-blur px-4 py-3 pl-5"
      style={{ opacity: c.cancelled ? 0.5 : 1, borderColor: c.cancelled ? 'rgba(251,191,36,.25)' : 'rgba(255,255,255,.08)', animation: 'popIn .35s ease both' }}
    >
      <span className="absolute left-2 top-3 bottom-3 w-[3px] rounded-full" style={{ background: c.accent, boxShadow: `0 0 8px ${c.accent}55` }} />
      <div className="text-[16px] font-semibold tracking-tight" style={{ color: c.cancelled ? '#a1a1aa' : '#fafafa', textDecoration: c.cancelled ? 'line-through' : 'none' }}>
        {c.title}
      </div>
      {c.detail && <div className="mono text-[11px] text-zinc-500 mt-1">{c.detail}</div>}
      <div className="flex items-center gap-1.5 mt-2.5">
        <span className="w-1.5 h-1.5 rounded-full" style={{ background: c.accent }} />
        <span className="mono text-[10px] tracking-wide" style={{ color: c.accent }}>{c.source}</span>
      </div>
    </div>
  )
}

export default function App() {
  const { state, send } = useAgentSocket()
  const { status, calMode, capMode, face, finals, interim, level, entities, cards, thinking, clarify, ttl, location } = state
  const [camOpen, setCamOpen] = useState(false)

  const st = STATUS[status] || STATUS.idle
  const faceMap = {
    present: { c: '#34d399', l: 'in view' },
    absent: { c: '#fb7185', l: 'no face' },
    off: { c: '#71717a', l: 'camera off' },
  }
  const fm = faceMap[face] || { c: '#71717a', l: 'face gate' }

  return (
    <div className="ambient h-full flex flex-col relative">
      {/* HEADER — brand · software states · hardware states */}
      <header className="grid grid-cols-3 items-center px-6 py-4 relative z-10">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1">
            <span className="w-4 h-4 rounded-full border-2 border-zinc-400" />
            <span className="w-1.5 h-[2px] bg-zinc-400" />
            <span className="w-4 h-4 rounded-full border-2 border-zinc-400" />
          </div>
          <div className="leading-none">
            <div className="text-[16px] font-semibold tracking-tight text-zinc-100">hearsay</div>
            <div className="mono text-[10px] tracking-[0.18em] text-zinc-600 mt-1">CALENDAR · ON GLASSES</div>
          </div>
        </div>

        <div className="flex items-center justify-center gap-2">
          <Chip color={calMode === 'live' ? '#34d399' : '#fbbf24'}>
            <span className={calMode === 'live' ? 'text-emerald-300/90' : 'text-amber-300/90'}>{calMode === 'live' ? 'Live' : 'Mock'}</span>
          </Chip>
          <Chip color="#34d399">{ttl > 0 ? `auto-deletes ${ttl}s` : 'ephemeral'}</Chip>
          {location && <Chip color="#38bdf8">📍 {location}</Chip>}
          <div className="flex items-center rounded-lg border border-white/10 overflow-hidden text-[13px]">
            {[['conversation', 'Conversation'], ['solo', 'Solo']].map(([m, label]) => (
              <button
                key={m}
                onClick={() => send('capturemode', { mode: m })}
                className={(capMode === m ? 'bg-white/10 text-white' : 'text-zinc-500 hover:text-zinc-300') + ' px-3 py-1.5 transition'}
              >
                {label}
              </button>
            ))}
          </div>
        </div>

        <div className="flex items-center justify-end gap-2">
          {capMode === 'conversation' && <Chip color={fm.c}>{fm.l}</Chip>}
          <span className="flex items-center gap-1.5 text-[12px] text-zinc-400 px-1">
            <Dot color={st.c} pulse={st.pulse} /> {st.l}
          </span>
          <Ctl onClick={() => send('mic')} title="Start microphone">Mic</Ctl>
          <Ctl onClick={() => setCamOpen((v) => !v)} title="Toggle face-mesh camera">Camera</Ctl>
          <Ctl onClick={() => send('reset')} title="Reset">↺</Ctl>
        </div>
      </header>

      {/* MAIN — two panels + connector */}
      <main className="flex-1 min-h-0 grid grid-cols-[1fr_64px_1.1fr] gap-0 px-6 pb-3 relative z-10">
        {/* LISTENING */}
        <section className="rounded-2xl border border-white/10 bg-zinc-900/40 backdrop-blur p-6 flex flex-col min-h-0 shadow-[0_20px_60px_-20px_rgba(0,0,0,0.7)]">
          <PanelHeader dot="#a1a1aa" title="Listening" sub="deepgram · nova-3" />
          <Waveform level={level} />
          {entities.length > 0 && (
            <div className="flex flex-wrap gap-1.5 mb-3">
              {entities.map((e, i) => (
                <span key={i} className="mono text-[10px] px-2 py-0.5 rounded-md bg-white/5 text-zinc-400 border border-white/10">{e}</span>
              ))}
            </div>
          )}
          <div className="flex-1 min-h-0 overflow-y-auto flex flex-col justify-end gap-2 mono text-[15px] leading-relaxed">
            {finals.length === 0 && !interim ? (
              <div className="m-auto text-zinc-600 text-sm">{capMode === 'solo' ? 'say “mark this, …”' : 'waiting for speech…'}</div>
            ) : (
              <>
                {finals.map((f, i) => {
                  const active = i === finals.length - 1 && !interim
                  return (
                    <div key={f.id} className={active ? 'text-zinc-100' : 'text-zinc-500'} style={{ animation: 'fadeUp .3s ease both' }}>{f.text}</div>
                  )
                })}
                {interim && <div className="text-zinc-100">{interim}</div>}
              </>
            )}
          </div>
        </section>

        {/* connector */}
        <div className="relative">
          <div className="absolute top-1/2 left-2 right-2 h-px -translate-y-px" style={{ background: 'linear-gradient(90deg,transparent,rgba(255,255,255,0.18),transparent)' }} />
          <span className="absolute top-1/2 right-1 w-1.5 h-1.5 rounded-full bg-zinc-500" style={{ transform: 'translateY(-50%)' }} />
        </div>

        {/* UNDERSTANDING */}
        <section className="rounded-2xl border border-white/10 bg-zinc-900/40 backdrop-blur p-6 flex flex-col min-h-0 shadow-[0_20px_60px_-20px_rgba(0,0,0,0.7)]">
          <PanelHeader dot="#818cf8" title="Understanding" sub="claude · haiku-4.5" />
          <div className="flex-1 min-h-0 flex flex-col">
            {clarify ? (
              <div className="flex-1 grid place-items-center">
                <div className="w-full max-w-md rounded-2xl border border-amber-400/30 bg-zinc-900/70 backdrop-blur p-5" style={{ animation: 'popIn .35s ease both' }}>
                  <div className="text-amber-200 font-medium mb-3">{clarify.question}</div>
                  <div className="flex flex-wrap gap-2">
                    {clarify.options.map((o, i) => (
                      <button key={i} onClick={() => send('answer', { text: o })} className="px-3 py-1.5 rounded-lg text-sm text-zinc-200 bg-white/5 border border-white/10 hover:border-amber-400/50 hover:text-white transition">{o}</button>
                    ))}
                  </div>
                </div>
              </div>
            ) : (cards.length || thinking.length) ? (
              <div className="flex-1 min-h-0 overflow-y-auto flex flex-col gap-2.5">
                {status === 'thinking' && (
                  <div className="flex items-center gap-2 text-zinc-500 text-[12px] mb-1">
                    <span className="w-1.5 h-1.5 rounded-full bg-indigo-400" style={{ animation: 'pulseDot 1s infinite' }} /> reading intent…
                  </div>
                )}
                {cards.map((c) => <ActionCard key={c.id} c={c} />)}
                {thinking.length > 0 && <RouterTrace lines={thinking} />}
              </div>
            ) : (
              <div className="flex-1 grid place-items-center">{status === 'thinking' ? <Thinking /> : <Idle />}</div>
            )}
          </div>
        </section>
      </main>

      {/* FOOTER */}
      <footer className="flex items-center justify-between px-6 py-3.5 relative z-10">
        <div className="mono text-[11px] text-zinc-600 flex items-center gap-2">
          <span>ray-ban meta mic</span><span className="text-zinc-700">→</span>
          <span>deepgram nova-3</span><span className="text-zinc-700">→</span>
          <span className="text-indigo-300/70">claude haiku-4.5</span><span className="text-zinc-700">→</span>
          <span className="text-emerald-300/70">your apps</span>
        </div>
        <span className="text-[13px] text-zinc-500">Your life, handled by your glasses.</span>
      </footer>

      <CameraMesh open={camOpen} onClose={() => setCamOpen(false)} />
    </div>
  )
}
