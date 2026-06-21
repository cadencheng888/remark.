import { useEffect, useReducer, useRef, useCallback } from 'react'
import { classifyAction } from './agent.js'

const TTL = 20
let cid = 0

const initial = {
  connected: false,
  status: 'connecting',     // listening | demo running | thinking | idle | disconnected
  calMode: 'mock',          // mock | live (calendar badge)
  location: null,           // detected current location label
  capMode: 'conversation',  // conversation | solo
  face: null,               // present | absent | off | null
  finals: [],               // [{id, text}]
  interim: '',
  level: 0,
  entities: [],             // recent entity strings (most-recent-first)
  cards: [],                // [{id, key, ...classified}]
  thinking: [],             // live agentic-router reasoning lines (the trace)
  clarify: null,            // {question, options}
  ttl: 0,                   // privacy countdown
}

function reducer(s, a) {
  switch (a.t) {
    case 'connected': return { ...s, connected: a.v, status: a.v ? 'idle' : 'disconnected' }
    case 'status': return { ...s, status: a.text, calMode: a.mode || s.calMode }
    case 'calMode': return { ...s, calMode: a.mode }
    case 'location': return { ...s, location: a.label }
    case 'capMode': return { ...s, capMode: a.mode, finals: [], interim: '', cards: [], thinking: [], clarify: null, entities: [] }
    case 'face': return { ...s, face: a.state }
    case 'level': return { ...s, level: a.value }
    case 'ttl': return { ...s, ttl: a.n }
    case 'caption':
      if (a.final) return { ...s, interim: '', finals: [...s.finals, { id: ++cid, text: a.text }].slice(-40) }
      return { ...s, interim: a.text }
    case 'entities': {
      const next = [...s.entities]
      for (const v of a.values) { if (v && !next.includes(v)) next.unshift(v) }
      return { ...s, entities: next.slice(0, 6) }
    }
    case 'thinking': {
      // The router's first trace line ('intent: "…"') marks a fresh run — start
      // a new reasoning group instead of appending to the previous one.
      const fresh = a.text.startsWith('intent:')
      const lines = fresh ? [a.text] : [...s.thinking, a.text]
      return { ...s, thinking: lines.slice(-16) }
    }
    case 'clarify': return { ...s, clarify: { question: a.question, options: a.options } }
    case 'action': {
      const c = classifyAction(a.text)
      if (a.muted || c.muted) return s
      // removal strikes the matching card already on the list
      if (c.cancelled && c.key) {
        const idx = s.cards.findIndex((x) => x.key === c.key && !x.cancelled)
        if (idx !== -1) {
          const cards = s.cards.slice()
          cards[idx] = { ...cards[idx], cancelled: true, accent: '#fbbf24', source: cards[idx].source === 'Calendar' ? 'cancelled' : 'removed' }
          return { ...s, cards, clarify: null }
        }
      }
      return { ...s, clarify: null, cards: [{ id: ++cid, ...c }, ...s.cards].slice(0, 6) }
    }
    case 'forget': return { ...s, finals: [], interim: '', entities: [], cards: [], thinking: [], clarify: null, ttl: 0 }
    case 'reset': return { ...initial, connected: s.connected, status: 'idle', calMode: s.calMode, capMode: s.capMode }
    default: return s
  }
}

export function useAgentSocket() {
  const [state, dispatch] = useReducer(reducer, initial)
  const wsRef = useRef(null)
  const ttlRef = useRef({ id: null, n: 0 })

  const bumpTTL = useCallback(() => {
    ttlRef.current.n = TTL
    dispatch({ t: 'ttl', n: TTL })
    if (!ttlRef.current.id) {
      ttlRef.current.id = setInterval(() => {
        ttlRef.current.n -= 1
        dispatch({ t: 'ttl', n: ttlRef.current.n })
        if (ttlRef.current.n <= 0) {
          clearInterval(ttlRef.current.id); ttlRef.current.id = null
          dispatch({ t: 'forget' })
        }
      }, 1000)
    }
  }, [])

  const clearTTL = useCallback(() => {
    if (ttlRef.current.id) { clearInterval(ttlRef.current.id); ttlRef.current.id = null }
    ttlRef.current.n = 0
  }, [])

  useEffect(() => {
    let stop = false
    const connect = () => {
      const proto = location.protocol === 'https:' ? 'wss://' : 'ws://'
      const ws = new WebSocket(proto + location.host + '/ws')
      wsRef.current = ws
      ws.onopen = () => dispatch({ t: 'connected', v: true })
      ws.onclose = () => { dispatch({ t: 'connected', v: false }); if (!stop) setTimeout(connect, 1500) }
      ws.onmessage = (e) => {
        const m = JSON.parse(e.data)
        switch (m.type) {
          case 'status': dispatch({ t: 'status', text: m.text, mode: m.mode }); break
          case 'capturemode': dispatch({ t: 'capMode', mode: m.mode }); break
          case 'caption': dispatch({ t: 'caption', text: m.text, final: m.final }); if (m.final) bumpTTL(); break
          case 'level': dispatch({ t: 'level', value: m.value }); break
          case 'entities': dispatch({ t: 'entities', values: m.values || [] }); break
          case 'action': dispatch({ t: 'action', text: m.text, muted: m.muted }); break
          case 'thinking': dispatch({ t: 'thinking', text: m.text }); break
          case 'clarify': dispatch({ t: 'clarify', question: m.question, options: m.options || [] }); break
          case 'forgotten': clearTTL(); dispatch({ t: 'forget' }); break
          case 'reset': clearTTL(); dispatch({ t: 'reset' }); break
          case 'face': dispatch({ t: 'face', state: m.state }); break
          default: break
        }
      }
    }
    connect()
    fetch('/config').then((r) => r.json()).then((c) => {
      dispatch({ t: 'calMode', mode: c.mode })
      if (c.location) dispatch({ t: 'location', label: c.location })
    }).catch(() => {})
    return () => { stop = true; clearTTL(); wsRef.current && wsRef.current.close() }
  }, [bumpTTL, clearTTL])

  const send = useCallback((cmd, extra = {}) => {
    const ws = wsRef.current
    if (ws && ws.readyState === 1) ws.send(JSON.stringify({ cmd, ...extra }))
  }, [])

  return { state, send }
}
