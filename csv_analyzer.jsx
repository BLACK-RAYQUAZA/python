import { useState, useCallback, useRef } from "react";

// ─── Papaparse-style CSV parser ────────────────────────────────────────────
function parseCSV(text) {
  const lines = text.trim().split(/\r?\n/);
  if (lines.length < 2) return { headers: [], rows: [] };
  const headers = lines[0].split(",").map(h => h.trim().replace(/^"|"$/g, ""));
  const rows = lines.slice(1).map(line => {
    const vals = line.split(",").map(v => v.trim().replace(/^"|"$/g, ""));
    const obj = {};
    headers.forEach((h, i) => { obj[h] = vals[i] ?? ""; });
    return obj;
  });
  return { headers, rows };
}

function isNumeric(col, rows) {
  return rows.slice(0, 20).some(r => r[col] !== "" && !isNaN(Number(r[col])));
}

function numVals(col, rows) {
  return rows.map(r => Number(r[col])).filter(v => !isNaN(v));
}

function stats(vals) {
  if (!vals.length) return null;
  const n = vals.length;
  const mean = vals.reduce((a, b) => a + b, 0) / n;
  const sorted = [...vals].sort((a, b) => a - b);
  const median = n % 2 === 0 ? (sorted[n/2-1]+sorted[n/2])/2 : sorted[Math.floor(n/2)];
  const std = Math.sqrt(vals.reduce((a, b) => a + (b - mean)**2, 0) / n);
  return { n, mean, median, std, min: sorted[0], max: sorted[n-1] };
}

function pearson(a, b) {
  const n = Math.min(a.length, b.length);
  if (n < 2) return 0;
  const ma = a.slice(0,n).reduce((s,x)=>s+x,0)/n;
  const mb = b.slice(0,n).reduce((s,x)=>s+x,0)/n;
  const num = a.slice(0,n).reduce((s,x,i)=>s+(x-ma)*(b[i]-mb),0);
  const den = Math.sqrt(a.slice(0,n).reduce((s,x)=>s+(x-ma)**2,0)*b.slice(0,n).reduce((s,x)=>s+(x-mb)**2,0));
  return den === 0 ? 0 : num/den;
}

// ─── Colors ────────────────────────────────────────────────────────────────
const C = {
  bg: "#0b0c10", card: "#13141a", border: "#22232e",
  accent: "#00d4aa", accent2: "#7c6af7", text: "#dde1f0", muted: "#565870",
  good: "#00d4aa", warn: "#f59e0b", danger: "#f43f5e",
};
const PALETTE = ["#00d4aa","#7c6af7","#f59e0b","#f43f5e","#38bdf8","#a3e635","#fb923c","#e879f9"];

const fmt = (v, isFloat) => isFloat ? Number(v.toFixed(3)).toLocaleString() : v.toLocaleString?.() ?? v;

// ─── SVG Bar ───────────────────────────────────────────────────────────────
function BarChart({ col, rows }) {
  const vals = numVals(col, rows);
  const buckets = 8;
  const mn = Math.min(...vals), mx = Math.max(...vals);
  const bw = (mx - mn) / buckets || 1;
  const bins = Array.from({length: buckets}, (_, i) => ({
    label: fmt(mn + i * bw, true),
    count: vals.filter(v => v >= mn+i*bw && v < mn+(i+1)*bw+(i===buckets-1?1:0)).length,
  }));
  const maxC = Math.max(...bins.map(b => b.count), 1);
  const W=460, H=180, pL=36, pB=32, pT=12, pR=8;
  const bWidth = (W-pL-pR)/buckets;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{width:"100%",height:"auto"}}>
      {[0,.25,.5,.75,1].map(t => {
        const y = pT+(H-pT-pB)*(1-t);
        return <g key={t}>
          <line x1={pL} y1={y} x2={W-pR} y2={y} stroke={C.border} strokeWidth="1"/>
          <text x={pL-4} y={y+3} fill={C.muted} fontSize="8" textAnchor="end">{Math.round(t*maxC)}</text>
        </g>;
      })}
      {bins.map((b,i) => {
        const bh = (b.count/maxC)*(H-pT-pB);
        const x = pL+i*bWidth+bWidth*0.1;
        return <g key={i}>
          <rect x={x} y={H-pB-bh} width={bWidth*0.8} height={bh}
            fill={C.accent} rx="2" opacity="0.85"/>
          <text x={x+bWidth*0.4} y={H-pB+10} fill={C.muted} fontSize="7" textAnchor="middle"
            transform={`rotate(-30,${x+bWidth*0.4},${H-pB+10})`}>{b.label}</text>
        </g>;
      })}
      <line x1={pL} y1={pT} x2={pL} y2={H-pB} stroke={C.border}/>
      <line x1={pL} y1={H-pB} x2={W-pR} y2={H-pB} stroke={C.border}/>
      <text x={W/2} y={H} fill={C.muted} fontSize="9" textAnchor="middle">{col} (distribution)</text>
    </svg>
  );
}

// ─── SVG Scatter ────────────────────────────────────────────────────────────
function ScatterPlot({ colX, colY, rows }) {
  const xs = numVals(colX, rows), ys = numVals(colY, rows);
  const n = Math.min(xs.length, ys.length, 200);
  const [mnX,mxX] = [Math.min(...xs)*0.97, Math.max(...xs)*1.03];
  const [mnY,mxY] = [Math.min(...ys)*0.97, Math.max(...ys)*1.03];
  const W=460, H=200, pL=40, pB=32, pT=12, pR=12;
  const px = v => pL+(v-mnX)/(mxX-mnX||1)*(W-pL-pR);
  const py = v => H-pB-(v-mnY)/(mxY-mnY||1)*(H-pT-pB);
  const r = pearson(xs.slice(0,n), ys.slice(0,n));
  const ma=xs.slice(0,n).reduce((a,b)=>a+b,0)/n, mb=ys.slice(0,n).reduce((a,b)=>a+b,0)/n;
  const slope = xs.slice(0,n).reduce((a,x,i)=>a+(x-ma)*(ys[i]-mb),0)/xs.slice(0,n).reduce((a,x)=>a+(x-ma)**2,1);
  const ic = mb - slope*ma;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{width:"100%",height:"auto"}}>
      {[0,.5,1].map(t => {
        const y=pT+(H-pT-pB)*t, x=pL+(W-pL-pR)*t;
        return <g key={t}>
          <line x1={pL} y1={y} x2={W-pR} y2={y} stroke={C.border} strokeWidth="1"/>
          <line x1={x} y1={pT} x2={x} y2={H-pB} stroke={C.border} strokeWidth="1"/>
        </g>;
      })}
      <line x1={px(mnX)} y1={py(slope*mnX+ic)} x2={px(mxX)} y2={py(slope*mxX+ic)}
        stroke={C.accent2} strokeWidth="1.5" strokeDasharray="5 3" opacity="0.8"/>
      {xs.slice(0,n).map((_,i) => (
        <circle key={i} cx={px(xs[i])} cy={py(ys[i])} r="3.5"
          fill={PALETTE[i%8]} opacity="0.75" stroke={C.bg} strokeWidth="1"/>
      ))}
      <line x1={pL} y1={pT} x2={pL} y2={H-pB} stroke={C.border}/>
      <line x1={pL} y1={H-pB} x2={W-pR} y2={H-pB} stroke={C.border}/>
      <text x={W-pR} y={pT+10} fill={C.accent2} fontSize="10" textAnchor="end" fontWeight="bold">r = {r.toFixed(3)}</text>
      <text x={(W+pL)/2} y={H} fill={C.muted} fontSize="9" textAnchor="middle">{colX}</text>
      <text x={8} y={H/2} fill={C.muted} fontSize="9" textAnchor="middle"
        transform={`rotate(-90,8,${H/2})`}>{colY}</text>
    </svg>
  );
}

// ─── SVG Heatmap ────────────────────────────────────────────────────────────
function Heatmap({ numCols, rows }) {
  const cols = numCols.slice(0, 6);
  const size = cols.length;
  if (size < 2) return <p style={{color:C.muted,fontSize:12}}>Need ≥ 2 numeric columns for heatmap.</p>;
  const cW=72, cH=52, pL=70, pT=36;
  const W=pL+cW*size+8, H=pT+cH*size+8;
  function heat(v) {
    if (v >= 0) { const t=v; return `rgb(${Math.round(0*t+19*(1-t))},${Math.round(212*t+20*(1-t))},${Math.round(170*t+26*(1-t)})`; }
    else { const t=-v; return `rgb(${Math.round(124*t+19*(1-t))},${Math.round(106*t+20*(1-t))},${Math.round(247*t+26*(1-t)})`; }
  }
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{width:"100%",maxWidth:500,height:"auto",display:"block",margin:"0 auto"}}>
      {cols.map((c,i) => (
        <text key={c} x={pL+i*cW+cW/2} y={pT-6} fill={C.text} fontSize="9" textAnchor="middle"
          style={{fontFamily:"monospace"}}>{c.length>8?c.slice(0,8)+"…":c}</text>
      ))}
      {cols.map((c,i) => (
        <text key={c} x={pL-4} y={pT+i*cH+cH/2+4} fill={C.text} fontSize="9" textAnchor="end"
          style={{fontFamily:"monospace"}}>{c.length>8?c.slice(0,8)+"…":c}</text>
      ))}
      {cols.map((ca,i) => cols.map((cb,j) => {
        const v = i===j ? 1 : pearson(numVals(ca,rows), numVals(cb,rows));
        return <g key={`${i}-${j}`}>
          <rect x={pL+j*cW} y={pT+i*cH} width={cW-2} height={cH-2} fill={heat(v)} rx="3"/>
          <text x={pL+j*cW+cW/2} y={pT+i*cH+cH/2+5}
            fill="#fff" fontSize="12" fontWeight="bold" textAnchor="middle">{v.toFixed(2)}</text>
        </g>;
      }))}
    </svg>
  );
}

// ─── Drop zone ──────────────────────────────────────────────────────────────
function DropZone({ onFile }) {
  const [drag, setDrag] = useState(false);
  const ref = useRef();
  const handle = (file) => {
    if (!file || !file.name.endsWith(".csv")) return;
    const reader = new FileReader();
    reader.onload = e => onFile(file.name, e.target.result);
    reader.readAsText(file);
  };
  return (
    <div
      onClick={() => ref.current.click()}
      onDragOver={e => { e.preventDefault(); setDrag(true); }}
      onDragLeave={() => setDrag(false)}
      onDrop={e => { e.preventDefault(); setDrag(false); handle(e.dataTransfer.files[0]); }}
      style={{
        border: `2px dashed ${drag ? C.accent : C.border}`,
        borderRadius: 16, padding: "56px 40px", textAlign: "center",
        cursor: "pointer", transition: "all 0.25s",
        background: drag ? "rgba(0,212,170,0.04)" : C.card,
      }}>
      <input ref={ref} type="file" accept=".csv" style={{display:"none"}}
        onChange={e => handle(e.target.files[0])}/>
      <div style={{fontSize:48,marginBottom:12}}>📂</div>
      <div style={{fontSize:18,fontWeight:800,color:C.text,marginBottom:6}}>Drop your CSV here</div>
      <div style={{fontSize:13,color:C.muted}}>or click to browse · any .csv file works</div>
    </div>
  );
}

// ─── Main ───────────────────────────────────────────────────────────────────
export default function App() {
  const [data, setData] = useState(null);
  const [filename, setFilename] = useState("");
  const [tab, setTab] = useState("overview");
  const [barCol, setBarCol] = useState("");
  const [scX, setScX] = useState(""), [scY, setScY] = useState("");

  const onFile = useCallback((name, text) => {
    const parsed = parseCSV(text);
    if (!parsed.rows.length) return;
    setData(parsed);
    setFilename(name);
    setTab("overview");
    const ncs = parsed.headers.filter(h => isNumeric(h, parsed.rows));
    setBarCol(ncs[0] || "");
    setScX(ncs[0] || ""); setScY(ncs[1] || ncs[0] || "");
  }, []);

  const btn = (id, label) => (
    <button key={id} onClick={() => setTab(id)} style={{
      padding:"7px 16px", borderRadius:6, border:"none", cursor:"pointer",
      fontSize:12, fontWeight:700, fontFamily:"inherit",
      background: tab===id ? C.accent : C.card,
      color: tab===id ? C.bg : C.muted,
      transition:"all 0.2s", letterSpacing:"0.04em",
    }}>{label}</button>
  );

  const sel = (val, setter, options) => (
    <select value={val} onChange={e=>setter(e.target.value)} style={{
      background:C.card, color:C.text, border:`1px solid ${C.border}`,
      borderRadius:6, padding:"5px 10px", fontSize:12, fontFamily:"inherit", cursor:"pointer",
    }}>
      {options.map(o => <option key={o} value={o}>{o}</option>)}
    </select>
  );

  const card = (children, extra={}) => (
    <div style={{background:C.card,border:`1px solid ${C.border}`,borderRadius:12,padding:20,...extra}}>
      {children}
    </div>
  );

  const label = (text) => (
    <div style={{fontSize:10,fontWeight:700,color:C.accent,letterSpacing:"0.1em",marginBottom:10}}>{text}</div>
  );

  if (!data) return (
    <div style={{minHeight:"100vh",background:C.bg,display:"flex",alignItems:"center",
      justifyContent:"center",padding:24,fontFamily:"'DM Mono','Courier New',monospace"}}>
      <div style={{width:"100%",maxWidth:520}}>
        <div style={{textAlign:"center",marginBottom:32}}>
          <div style={{fontSize:13,color:C.accent,letterSpacing:"0.15em",fontWeight:700,marginBottom:8}}>
            COLLEGE PROJECT
          </div>
          <h1 style={{margin:0,fontSize:28,fontWeight:900,color:C.text,letterSpacing:"-0.03em"}}>
            CSV Data Analyser
          </h1>
          <p style={{color:C.muted,fontSize:13,marginTop:8}}>
            Pandas · Matplotlib · Statistical Analysis
          </p>
        </div>
        <DropZone onFile={onFile}/>
        <p style={{textAlign:"center",color:C.muted,fontSize:11,marginTop:16}}>
          Supports any well-formed CSV · data never leaves your browser
        </p>
      </div>
    </div>
  );

  const { headers, rows } = data;
  const numCols = headers.filter(h => isNumeric(h, rows));
  const catCols = headers.filter(h => !isNumeric(h, rows));

  // value counts for first categorical col
  const catCol = catCols[0];
  const valCounts = catCol ? Object.entries(
    rows.reduce((acc,r)=>{ const v=r[catCol]||"(blank)"; acc[v]=(acc[v]||0)+1; return acc; },{}))
    .sort((a,b)=>b[1]-a[1]).slice(0,10) : [];

  const W_vc=460, H_vc=180, pL_vc=80, pB_vc=20, pT_vc=10, pR_vc=8;
  const maxVC = valCounts[0]?.[1] || 1;

  return (
    <div style={{minHeight:"100vh",background:C.bg,color:C.text,
      fontFamily:"'DM Mono','Courier New',monospace",padding:"20px 12px"}}>

      {/* Header */}
      <div style={{maxWidth:900,margin:"0 auto 20px"}}>
        <div style={{display:"flex",alignItems:"center",justifyContent:"space-between",flexWrap:"wrap",gap:8}}>
          <div>
            <span style={{fontSize:10,color:C.accent,letterSpacing:"0.12em",fontWeight:700}}>LOADED ✓</span>
            <h2 style={{margin:"2px 0 0",fontSize:18,fontWeight:900,color:C.text}}>{filename}</h2>
            <span style={{fontSize:11,color:C.muted}}>{rows.length} rows × {headers.length} columns
              · {numCols.length} numeric · {catCols.length} categorical</span>
          </div>
          <button onClick={()=>setData(null)} style={{
            padding:"8px 16px",borderRadius:8,border:`1px solid ${C.border}`,
            background:"transparent",color:C.muted,cursor:"pointer",fontSize:12,fontFamily:"inherit",
          }}>↩ New file</button>
        </div>

        {/* KPIs */}
        <div style={{display:"grid",gridTemplateColumns:"repeat(auto-fill,minmax(160px,1fr))",gap:8,marginTop:14}}>
          {numCols.slice(0,4).map(c => {
            const s = stats(numVals(c, rows));
            return s ? (
              <div key={c} style={{background:C.card,border:`1px solid ${C.border}`,borderRadius:10,padding:"12px 14px"}}>
                <div style={{fontSize:10,color:C.muted,marginBottom:4,overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap"}}>{c}</div>
                <div style={{fontSize:17,fontWeight:900,color:C.accent}}>{fmt(s.mean, true)}</div>
                <div style={{fontSize:9,color:C.muted}}>mean · σ {fmt(s.std,true)}</div>
              </div>
            ) : null;
          })}
        </div>
      </div>

      <div style={{maxWidth:900,margin:"0 auto"}}>
        {/* Tabs */}
        <div style={{display:"flex",gap:6,flexWrap:"wrap",marginBottom:14}}>
          {[["overview","Overview"],["bar","Distribution"],["scatter","Scatter"],["heatmap","Heatmap"],["table","Raw Data"]].map(([id,lbl])=>btn(id,lbl))}
        </div>

        {/* ── Overview ── */}
        {tab==="overview" && (
          <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:12}}>
            {card(<>
              {label("df.describe() — NUMERIC COLUMNS")}
              <div style={{overflowX:"auto"}}>
                <table style={{width:"100%",borderCollapse:"collapse",fontSize:11}}>
                  <thead>
                    <tr>{["column","count","mean","median","std","min","max"].map(h=>(
                      <th key={h} style={{padding:"5px 8px",textAlign:"right",color:C.accent,
                        borderBottom:`1px solid ${C.border}`,fontSize:9,letterSpacing:"0.06em",
                        whiteSpace:"nowrap"}}>{h}</th>
                    ))}</tr>
                  </thead>
                  <tbody>
                    {numCols.map(c => {
                      const s = stats(numVals(c, rows));
                      return s ? (
                        <tr key={c} style={{borderBottom:`1px solid ${C.border}`}}>
                          <td style={{padding:"5px 8px",color:C.accent,fontWeight:700,fontSize:10,maxWidth:80,overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap"}}>{c}</td>
                          {[s.n,s.mean,s.median,s.std,s.min,s.max].map((v,i)=>(
                            <td key={i} style={{padding:"5px 8px",textAlign:"right",color:C.text,fontSize:10}}>{fmt(v,true)}</td>
                          ))}
                        </tr>
                      ) : null;
                    })}
                  </tbody>
                </table>
              </div>
            </>)}

            {card(<>
              {label("VALUE COUNTS — " + (catCol||"(no categorical)"))}
              {catCol ? (
                <svg viewBox={`0 0 ${W_vc} ${H_vc}`} style={{width:"100%",height:"auto"}}>
                  {valCounts.map(([v,c],i)=>{
                    const bw=(W_vc-pL_vc-pR_vc)/valCounts.length;
                    const bh=(c/maxVC)*(H_vc-pT_vc-pB_vc);
                    const x=pL_vc+i*bw+bw*0.1;
                    return <g key={v}>
                      <rect x={x} y={H_vc-pB_vc-bh} width={bw*0.8} height={bh}
                        fill={PALETTE[i%8]} rx="2" opacity="0.85"/>
                      <text x={x+bw*0.4} y={H_vc-pB_vc+12} fill={C.muted} fontSize="8"
                        textAnchor="middle" transform={`rotate(-35,${x+bw*0.4},${H_vc-pB_vc+12})`}>
                        {v.length>8?v.slice(0,8)+"…":v}
                      </text>
                    </g>;
                  })}
                  <line x1={pL_vc} y1={pT_vc} x2={pL_vc} y2={H_vc-pB_vc} stroke={C.border}/>
                  <line x1={pL_vc} y1={H_vc-pB_vc} x2={W_vc-pR_vc} y2={H_vc-pB_vc} stroke={C.border}/>
                </svg>
              ) : <p style={{color:C.muted,fontSize:12}}>No categorical columns found.</p>}
            </>)}

            {card(<>
              {label("df.isnull().sum() — MISSING VALUES")}
              {headers.map(h => {
                const missing = rows.filter(r => !r[h] || r[h]==="").length;
                const pct = (missing/rows.length*100).toFixed(1);
                return (
                  <div key={h} style={{marginBottom:6}}>
                    <div style={{display:"flex",justifyContent:"space-between",fontSize:10,marginBottom:2}}>
                      <span style={{color:C.muted,maxWidth:160,overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap"}}>{h}</span>
                      <span style={{color:missing>0?C.warn:C.good,fontWeight:700}}>{missing} ({pct}%)</span>
                    </div>
                    <div style={{height:3,background:C.border,borderRadius:2}}>
                      <div style={{height:"100%",width:`${pct}%`,background:missing>0?C.warn:C.good,borderRadius:2}}/>
                    </div>
                  </div>
                );
              })}
            </>, {gridColumn:"1/-1"})}
          </div>
        )}

        {/* ── Distribution Bar ── */}
        {tab==="bar" && card(<>
          {label("HISTOGRAM / DISTRIBUTION")}
          <div style={{display:"flex",alignItems:"center",gap:8,marginBottom:14}}>
            <span style={{fontSize:11,color:C.muted}}>Column:</span>
            {sel(barCol, setBarCol, numCols)}
          </div>
          {barCol ? <BarChart col={barCol} rows={rows}/> : <p style={{color:C.muted}}>No numeric columns.</p>}
        </>)}

        {/* ── Scatter ── */}
        {tab==="scatter" && card(<>
          {label("SCATTER PLOT + REGRESSION LINE")}
          <div style={{display:"flex",alignItems:"center",gap:8,flexWrap:"wrap",marginBottom:14}}>
            <span style={{fontSize:11,color:C.muted}}>X:</span>{sel(scX,setScX,numCols)}
            <span style={{fontSize:11,color:C.muted}}>Y:</span>{sel(scY,setScY,numCols)}
          </div>
          {scX && scY ? <ScatterPlot colX={scX} colY={scY} rows={rows}/> : <p style={{color:C.muted}}>Select columns.</p>}
          <div style={{marginTop:10,fontSize:11,color:C.muted}}>
            Pearson r = <span style={{color:C.accent2,fontWeight:700}}>
              {pearson(numVals(scX,rows),numVals(scY,rows)).toFixed(4)}</span>
            &nbsp;· Dashed line = linear regression
          </div>
        </>)}

        {/* ── Heatmap ── */}
        {tab==="heatmap" && card(<>
          {label("CORRELATION MATRIX HEATMAP")}
          <p style={{fontSize:11,color:C.muted,marginTop:0,marginBottom:14}}>
            <span style={{color:C.accent}}>■ teal = positive</span> &nbsp;
            <span style={{color:C.accent2}}>■ purple = negative</span> &nbsp;
            (first 6 numeric columns)
          </p>
          <Heatmap numCols={numCols} rows={rows}/>
        </>)}

        {/* ── Raw table ── */}
        {tab==="table" && card(<>
          {label(`RAW DATA — FIRST 50 ROWS (of ${rows.length})`)}
          <div style={{overflowX:"auto"}}>
            <table style={{width:"100%",borderCollapse:"collapse",fontSize:11,whiteSpace:"nowrap"}}>
              <thead>
                <tr>{headers.map(h=>(
                  <th key={h} style={{padding:"6px 10px",textAlign:"left",color:C.accent,
                    borderBottom:`1px solid ${C.border}`,fontSize:9,letterSpacing:"0.06em"}}>{h}</th>
                ))}</tr>
              </thead>
              <tbody>
                {rows.slice(0,50).map((row,i)=>(
                  <tr key={i} style={{borderBottom:`1px solid ${C.border}`,
                    background:i%2===0?"transparent":"rgba(255,255,255,0.015)"}}>
                    {headers.map(h=>(
                      <td key={h} style={{padding:"5px 10px",color:isNumeric(h,rows)?C.accent:C.text}}>
                        {row[h]}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>)}
      </div>
    </div>
  );
}
