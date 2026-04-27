import { useState, useEffect } from "react";

// Backend API endpoint — FastAPI running on AWS EC2 port 8000
const API_URL = "http://3.86.94.48:8000";

// Dropdown options matching the ML model's training categories
const INDUSTRIES     = ["Manufacturing","SaaS","Retail","Healthcare","Finance"];
const DECISION_TYPES = ["Marketing","Pricing","Expansion","RD_Investment","Hiring"];
const SIZES          = ["Small","Medium","Large"];
const INTENSITIES    = ["Low","Medium","High"];

// Default form values used on initial page load
const defaults = {
  Industry:"Manufacturing", Company_Size:"Small",
  Operating_Margin:0.08, Baseline_Revenue:100000000,
  Decision_Type:"Marketing", Investment_Cost:2000000,
  Time_Horizon_Months:6, Campaign_Intensity:"High",
  GDP_Growth:-0.02, Inflation_Expect:0.04,
  Unemployment:0.065, Interest_Rate:0.05,
  VIX:35, Vol_30D:0.04, Mkt_Ret:-0.01,
  External_Risk_Flag:1, news_text:"", anthropic_api_key:"",
};

// Deloitte brand green — used throughout the UI
const G = "#86BC25"; // Deloitte green

// Global CSS injected into the document head
const styles = `
  *,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
  body{background:#f4f4f4;font-family:'Arial',sans-serif;color:#1a1a1a;min-height:100vh}
  ::-webkit-scrollbar{width:4px}
  ::-webkit-scrollbar-track{background:#f4f4f4}
  ::-webkit-scrollbar-thumb{background:#ddd;border-radius:2px}
  select option{background:#fff;color:#1a1a1a}
  @keyframes fadeIn{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}
  @keyframes spin{to{transform:rotate(360deg)}}
`;

// Reusable uppercase label component for form fields
function FieldLabel({ children }) {
  return (
    <div style={{ fontSize:10, fontWeight:600, letterSpacing:0.5, color:"#666",
      marginBottom:5, textTransform:"uppercase" }}>
      {children}
    </div>
  );
}

// Numeric/text input with animated Deloitte green focus border
function InputField({ label, value, onChange, type="number", step }) {
  const [f, setF] = useState(false);
  return (
    <div style={{ marginBottom:13 }}>
      <FieldLabel>{label}</FieldLabel>
      <input type={type} value={value} step={step}
        onChange={e => onChange(type==="number" ? parseFloat(e.target.value)||0 : e.target.value)}
        onFocus={() => setF(true)} onBlur={() => setF(false)}
        style={{
          width:"100%", background:"#fff", border:"none",
          borderBottom:`1.5px solid ${f ? G : "#ddd"}`,
          padding:"7px 0", fontSize:13, color:"#1a1a1a",
          outline:"none", transition:"border-color .2s", fontFamily:"Arial,sans-serif",
        }}
      />
    </div>
  );
}

// Dropdown selector with consistent styling
function SelectField({ label, value, onChange, options }) {
  const [f, setF] = useState(false);
  return (
    <div style={{ marginBottom:13 }}>
      <FieldLabel>{label}</FieldLabel>
      <select value={value} onChange={e => onChange(e.target.value)}
        onFocus={() => setF(true)} onBlur={() => setF(false)}
        style={{
          width:"100%", background:"#fff", border:"none",
          borderBottom:`1.5px solid ${f ? G : "#ddd"}`,
          padding:"7px 0", fontSize:13, color:"#1a1a1a",
          outline:"none", cursor:"pointer", appearance:"none",
          backgroundImage:`url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='8' height='5'%3E%3Cpath d='M0 0l4 5 4-5z' fill='%23999'/%3E%3C/svg%3E")`,
          backgroundRepeat:"no-repeat", backgroundPosition:"right 2px center",
          paddingRight:16, fontFamily:"Arial,sans-serif",
        }}
      >
        {options.map(o => <option key={String(o)} value={o}>{String(o)}</option>)}
      </select>
    </div>
  );
}

function SectionHead({ children }) {
  return (
    <div style={{ fontSize:9, fontWeight:700, letterSpacing:2, textTransform:"uppercase",
      color:G, marginBottom:12, paddingBottom:5, borderBottom:`2px solid ${G}`,
      display:"inline-block" }}>
      {children}
    </div>
  );
}

function Divider() {
  return <div style={{ height:1, background:"#e8e8e8", margin:"16px 0" }} />;
}

export default function App() {
  const [form, setForm]     = useState(defaults);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]   = useState("");
  const [clock, setClock]   = useState("");

  useEffect(() => {
    const tick = () => {
      const n = new Date();
      const p = x => String(x).padStart(2,"0");
      setClock(`${p(n.getHours())}:${p(n.getMinutes())}:${p(n.getSeconds())} EST`);
    };
    tick();
    const t = setInterval(tick, 1000);
    return () => clearInterval(t);
  }, []);

  const set = k => v => setForm(f => ({ ...f, [k]: v }));

  const submit = async () => {
    setLoading(true); setError(""); setResult(null);
    try {
      const r = await fetch(`${API_URL}/predict`, {
        method:"POST", headers:{"Content-Type":"application/json"},
        body: JSON.stringify(form),
      });
      if (!r.ok) throw new Error(`Server error ${r.status}`);
      setResult(await r.json());
    } catch(e) { setError(e.message); }
    finally { setLoading(false); }
  };

  return (
    <>
      <style>{styles}</style>

      {/* HEADER */}
      <div style={{ background:"#000", height:56, display:"flex", alignItems:"center",
        justifyContent:"space-between", padding:"0 28px" }}>
        <div style={{ display:"flex", alignItems:"center", gap:20 }}>
          <div style={{ display:"flex", alignItems:"center", gap:3 }}>
            <span style={{ fontSize:20, fontWeight:900, color:"#fff", letterSpacing:-0.5 }}>Deloitte</span>
            <div style={{ width:6, height:6, background:G, borderRadius:"50%",
              marginBottom:10, marginLeft:1 }} />
          </div>
          <div style={{ width:1, height:28, background:"#333" }} />
          <div>
            <div style={{ fontSize:12, fontWeight:600, color:"#fff", letterSpacing:0.5 }}>
              Decision Intelligence Platform
            </div>
            <div style={{ fontSize:10, color:G, letterSpacing:1, fontFamily:"'Courier New',monospace" }}>
              ROI PREDICTION ENGINE · AWS + CLAUDE AI
            </div>
          </div>
        </div>
        <div style={{ display:"flex", alignItems:"center", gap:20 }}>
          <div style={{ display:"flex", alignItems:"center", gap:7 }}>
            <div style={{ width:7, height:7, borderRadius:"50%", background:G }} />
            <span style={{ fontSize:10, color:G, fontWeight:600, letterSpacing:1 }}>LIVE</span>
          </div>
          <div style={{ width:1, height:24, background:"#333" }} />
          <span style={{ fontFamily:"'Courier New',monospace", fontSize:12, color:G, fontWeight:600 }}>
            {clock}
          </span>
          <div style={{ width:32, height:32, borderRadius:"50%", background:G,
            display:"grid", placeItems:"center" }}>
            <span style={{ fontSize:13, fontWeight:900, color:"#000" }}>K</span>
          </div>
        </div>
      </div>

      {/* STATS BAR */}
      <div style={{ background:"#1a1a1a", padding:"0 28px", display:"flex",
        alignItems:"center", height:38, borderBottom:`3px solid ${G}` }}>
        {[
          ["Training Corpus","120,000","#fff"],
          ["Directional Accuracy","75.2%",G],
          ["Industries","5","#fff"],
          ["Ensemble","GBM · XGBoost · LightGBM","#fff"],
          ["AI Layer","Claude AI",G],
        ].map(([label, val, color], i, arr) => (
          <div key={label} style={{ display:"flex", alignItems:"center", gap:6,
            padding:`0 ${i===0?"0 22px":"22px"}`,
            borderRight: i < arr.length-1 ? "1px solid #333" : "none", height:"100%" }}>
            <span style={{ fontSize:10, color:"#888" }}>{label}</span>
            <span style={{ fontSize:10, fontWeight:700, color }}>{val}</span>
          </div>
        ))}
      </div>

      {/* BODY */}
      <div style={{ display:"flex", height:"calc(100vh - 94px)" }}>

        {/* SIDEBAR */}
        <div style={{ width:250, flexShrink:0, background:"#fff",
          borderRight:"1px solid #e0e0e0", padding:"20px 18px", overflowY:"auto" }}>
          <SectionHead>Company</SectionHead>
          <SelectField label="Industry"     value={form.Industry}     onChange={set("Industry")}     options={INDUSTRIES} />
          <SelectField label="Company Size" value={form.Company_Size} onChange={set("Company_Size")} options={SIZES} />
          <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:12, marginBottom:13 }}>
            <InputField label="Revenue ($M)" value={form.Baseline_Revenue/1e6} onChange={v => set("Baseline_Revenue")(v*1e6)} step={1} />
            <InputField label="Op. Margin"   value={form.Operating_Margin}     onChange={set("Operating_Margin")} step={0.01} />
          </div>
          <Divider />
          <SectionHead>Decision</SectionHead>
          <SelectField label="Decision Type"     value={form.Decision_Type}     onChange={set("Decision_Type")}     options={DECISION_TYPES} />
          <SelectField label="Campaign Intensity" value={form.Campaign_Intensity} onChange={set("Campaign_Intensity")} options={INTENSITIES} />
          <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:12, marginBottom:13 }}>
            <InputField label="Investment ($M)" value={form.Investment_Cost/1e6} onChange={v => set("Investment_Cost")(v*1e6)} step={0.5} />
            <InputField label="Horizon (Mo)"   value={form.Time_Horizon_Months} onChange={set("Time_Horizon_Months")} step={1} />
          </div>
          <Divider />
          <SectionHead>Macro</SectionHead>
          <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:12, marginBottom:13 }}>
            <InputField label="GDP Growth" value={form.GDP_Growth}   onChange={set("GDP_Growth")}   step={0.001} />
            <InputField label="VIX Index"  value={form.VIX}          onChange={set("VIX")}          step={1} />
          </div>
          <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:12, marginBottom:13 }}>
            <InputField label="Unemployment" value={form.Unemployment}  onChange={set("Unemployment")}  step={0.001} />
            <InputField label="Interest Rate" value={form.Interest_Rate} onChange={set("Interest_Rate")} step={0.001} />
          </div>
          <SelectField label="External Risk"
            value={form.External_Risk_Flag}
            onChange={v => set("External_Risk_Flag")(parseInt(v))}
            options={[0,1]}
          />
        </div>

        {/* MAIN */}
        <div style={{ flex:1, display:"flex", flexDirection:"column", overflow:"hidden", minWidth:0 }}>

          {/* Input area */}
          <div style={{ background:"#fff", borderBottom:"1px solid #e0e0e0",
            padding:"16px 22px", flexShrink:0 }}>
            <div style={{ display:"flex", gap:18, alignItems:"flex-start" }}>
              <div style={{ flex:1, minWidth:0 }}>
                <div style={{ display:"flex", alignItems:"center", gap:10, marginBottom:8 }}>
                  <span style={{ fontSize:13, fontWeight:700, color:"#1a1a1a" }}>
                    Market Intelligence Input
                  </span>
                  <span style={{ fontSize:9, fontWeight:700, color:"#fff", background:G,
                    padding:"2px 8px", borderRadius:2, letterSpacing:1, whiteSpace:"nowrap" }}>
                    CLAUDE AI
                  </span>
                </div>
                <textarea value={form.news_text} onChange={e => set("news_text")(e.target.value)}
                  rows={3}
                  placeholder="Paste market news or analyst reports. Claude AI extracts consumer sentiment, competitive pressure, and external risk — merged as model features before inference..."
                  style={{
                    width:"100%", background:"#f9f9f9", border:"1.5px solid #e0e0e0",
                    borderRadius:3, padding:"10px 12px", fontSize:12.5, color:"#333",
                    lineHeight:1.6, resize:"none", outline:"none",
                    fontFamily:"Arial,sans-serif", boxSizing:"border-box",
                  }}
                  onFocus={e => e.target.style.borderColor=G}
                  onBlur={e => e.target.style.borderColor="#e0e0e0"}
                />
              </div>
              <div style={{ width:185, flexShrink:0 }}>
                <FieldLabel>Anthropic API Key</FieldLabel>
                <input type="password" value={form.anthropic_api_key}
                  onChange={e => set("anthropic_api_key")(e.target.value)}
                  placeholder="sk-ant-..."
                  style={{
                    width:"100%", background:"#f9f9f9", border:"1.5px solid #e0e0e0",
                    borderRadius:3, padding:"8px 10px", fontSize:12, color:"#1a1a1a",
                    outline:"none", marginBottom:10,
                    fontFamily:"'Courier New',monospace", boxSizing:"border-box",
                  }}
                />
                <button onClick={submit} disabled={loading}
                  style={{
                    width:"100%", padding:"13px 0", background: loading ? "#333" : "#000",
                    color:G, border:"none", borderRadius:3, fontSize:11,
                    fontWeight:700, letterSpacing:2, cursor: loading ? "not-allowed" : "pointer",
                    textTransform:"uppercase", fontFamily:"Arial,sans-serif",
                  }}>
                  {loading ? "PROCESSING..." : "▶  RUN ANALYSIS"}
                </button>
              </div>
            </div>
          </div>

          {/* Results scroll area */}
          <div style={{ flex:1, overflowY:"auto", padding:"16px 22px",
            background:"#f4f4f4", minWidth:0 }}>

            {error && (
              <div style={{ background:"#fff0f0", border:"1px solid #ffcccc",
                borderRadius:4, padding:"12px 16px", marginBottom:14,
                color:"#cc0000", fontSize:12 }}>
                ⚠ {error}
              </div>
            )}

            {loading && (
              <div style={{ display:"flex", alignItems:"center", justifyContent:"center",
                gap:16, padding:"80px 0" }}>
                <div style={{ width:28, height:28, border:"2px solid #e0e0e0",
                  borderTop:`2px solid ${G}`, borderRadius:"50%",
                  animation:"spin 0.8s linear infinite" }} />
                <div>
                  <div style={{ fontSize:15, color:"#1a1a1a" }}>Analyzing Decision Parameters</div>
                  <div style={{ fontSize:11, color:"#999", marginTop:3 }}>
                    Running ensemble model + AI signal extraction...
                  </div>
                </div>
              </div>
            )}

            {!result && !loading && !error && (
              <div style={{ height:260, display:"flex", flexDirection:"column",
                alignItems:"center", justifyContent:"center", textAlign:"center" }}>
                <div style={{ width:44, height:44, border:"2px solid #ddd",
                  borderRadius:"50%", display:"grid", placeItems:"center", marginBottom:14 }}>
                  <div style={{ width:16, height:16, border:"2px solid #ccc", borderRadius:"50%" }} />
                </div>
                <div style={{ fontSize:15, color:"#999" }}>Awaiting analysis parameters</div>
                <div style={{ fontSize:10, color:G, fontWeight:700,
                  letterSpacing:2, marginTop:6 }}>CONFIGURE AND RUN</div>
              </div>
            )}

            {result && !loading && (
              <div style={{ animation:"fadeIn .4s ease" }}>

                {/* KPI STRIP */}
                <div style={{ display:"grid", gridTemplateColumns:"repeat(4,1fr)",
                  borderRadius:4, overflow:"hidden", marginBottom:14,
                  border:"1.5px solid #000", height:110 }}>

                  {/* ROI */}
                  <div style={{ background:"#000", padding:"14px 18px",
                    display:"flex", flexDirection:"column", justifyContent:"space-between",
                    borderRight:"1px solid #222", minWidth:0 }}>
                    <div style={{ fontSize:9, fontWeight:700, color:G,
                      letterSpacing:1.5, whiteSpace:"nowrap" }}>PREDICTED ROI</div>
                    <div style={{ fontSize:34, fontWeight:900, color:G,
                      lineHeight:1, whiteSpace:"nowrap" }}>
                      {result.roi >= 0 ? "+" : ""}{(result.roi*100).toFixed(1)}%
                    </div>
                    <div style={{ fontSize:9, color:G, opacity:0.7,
                      fontWeight:600, whiteSpace:"nowrap" }}>
                      {result.roi > 0 ? "▲ Above industry avg" : "▼ Below industry avg"}
                    </div>
                  </div>

                  {/* Confidence */}
                  <div style={{ background:"#fff", padding:"14px 18px",
                    display:"flex", flexDirection:"column", justifyContent:"space-between",
                    borderRight:"1px solid #e0e0e0", minWidth:0 }}>
                    <div style={{ fontSize:9, fontWeight:700, color:"#666",
                      letterSpacing:1.5, whiteSpace:"nowrap" }}>AI CONFIDENCE</div>
                    <div style={{ fontSize:34, fontWeight:900, color:"#1a1a1a", lineHeight:1 }}>
                      {(result.confidence*100).toFixed(0)}%
                    </div>
                    <div style={{ height:4, background:"#f0f0f0", borderRadius:2 }}>
                      <div style={{ height:"100%", width:`${result.confidence*100}%`,
                        background:G, borderRadius:2 }} />
                    </div>
                  </div>

                  {/* Risk */}
                  <div style={{ background:"#fff", padding:"14px 18px",
                    display:"flex", flexDirection:"column", justifyContent:"space-between",
                    borderRight:"1px solid #e0e0e0", minWidth:0 }}>
                    <div style={{ fontSize:9, fontWeight:700, color:"#666",
                      letterSpacing:1.5, whiteSpace:"nowrap" }}>RISK LEVEL</div>
                    <div style={{ fontSize:34, fontWeight:900, color:"#1a1a1a",
                      lineHeight:1, whiteSpace:"nowrap" }}>{result.risk_level}</div>
                    <div style={{ display:"flex", gap:3 }}>
                      {["#86BC25","#c8e06a","#f59e0b","#eee"].map((c,i) => (
                        <div key={i} style={{ flex:1, height:4, background:c, borderRadius:1 }} />
                      ))}
                    </div>
                  </div>

                  {/* Verdict */}
                  <div style={{ background:"#fff", padding:"14px 18px",
                    display:"flex", flexDirection:"column", justifyContent:"space-between",
                    minWidth:0 }}>
                    <div style={{ fontSize:9, fontWeight:700, color:"#666",
                      letterSpacing:1.5, whiteSpace:"nowrap" }}>VERDICT</div>
                    <div style={{ fontSize:18, fontWeight:900, color:"#1a1a1a",
                      lineHeight:1.2 }}>
                      {result.roi > 0.2 ? "Proceed" : result.roi > 0 ? "Proceed" : "Do Not"}
                      <br />
                      {result.roi > 0 ? "with Caution" : "Proceed"}
                    </div>
                    <div style={{ display:"inline-block", fontSize:9, fontWeight:700,
                      color:G, background:"#f0f7e6", padding:"2px 8px",
                      borderRadius:2, letterSpacing:1, whiteSpace:"nowrap" }}>
                      {result.roi > 0.2 ? "PHASE INVESTMENT" :
                       result.roi > 0   ? "REDUCE EXPOSURE"  : "HOLD DECISION"}
                    </div>
                  </div>
                </div>

                {/* Signals + Report */}
                <div style={{ display:"grid", gridTemplateColumns:"195px 1fr",
                  gap:13, minWidth:0 }}>

                  {/* Signals */}
                  <div style={{ background:"#fff", border:"1px solid #e0e0e0",
                    borderRadius:4, padding:16, minWidth:0 }}>
                    <SectionHead>AI Signals</SectionHead>
                    {[
                      { label:"SENTIMENT",   val:result.consumer_sentiment.toFixed(2),
                        pct: ((result.consumer_sentiment+1)/2*100), color:"#1a1a1a",
                        note: result.consumer_sentiment < -0.3 ? "Bearish outlook" : "Neutral outlook" },
                      { label:"COMPETITION", val:result.competitive_pressure.toFixed(2),
                        pct: result.competitive_pressure*100, color:G,
                        note: result.competitive_pressure > 0.5 ? "Moderate pressure" : "Low pressure" },
                    ].map(({ label, val, pct, color, note }) => (
                      <div key={label} style={{ marginBottom:13 }}>
                        <div style={{ display:"flex", justifyContent:"space-between",
                          alignItems:"center", marginBottom:5 }}>
                          <span style={{ fontSize:10, fontWeight:600, color:"#666" }}>{label}</span>
                          <span style={{ fontSize:14, fontWeight:800, color:"#1a1a1a" }}>{val}</span>
                        </div>
                        <div style={{ height:4, background:"#f0f0f0", borderRadius:2 }}>
                          <div style={{ height:"100%", width:`${pct}%`,
                            background:color, borderRadius:2 }} />
                        </div>
                        <div style={{ fontSize:10, color:"#999", marginTop:3 }}>{note}</div>
                      </div>
                    ))}
                    <div style={{ marginBottom:13 }}>
                      <div style={{ display:"flex", justifyContent:"space-between",
                        alignItems:"center", marginBottom:5 }}>
                        <span style={{ fontSize:10, fontWeight:600, color:"#666" }}>EXT. RISK</span>
                        <span style={{ fontSize:10, fontWeight:700, color:"#fff",
                          background:"#1a1a1a", padding:"2px 7px", borderRadius:2 }}>
                          {result.external_risk ? "FLAGGED" : "CLEAR"}
                        </span>
                      </div>
                      <div style={{ height:4, background:"#f0f0f0", borderRadius:2 }}>
                        <div style={{ height:"100%",
                          width: result.external_risk ? "100%" : "0%",
                          background:"#1a1a1a", borderRadius:2 }} />
                      </div>
                    </div>
                    <Divider />
                    <SectionHead>Model Stack</SectionHead>
                    {[["Gradient Boosting","GBM"],["Extreme Gradient","XGBoost"],["Light Gradient","LightGBM"]].map(([n,t])=>(
                      <div key={t} style={{ display:"flex", justifyContent:"space-between",
                        alignItems:"center", marginBottom:7 }}>
                        <span style={{ fontSize:11, color:"#666" }}>{n}</span>
                        <span style={{ fontSize:9, fontWeight:700, color:G,
                          background:"#f0f7e6", padding:"2px 6px", borderRadius:2 }}>{t}</span>
                      </div>
                    ))}
                    <div style={{ marginTop:11, background:"#000", borderRadius:3, padding:"8px 10px" }}>
                      <div style={{ fontSize:9, fontWeight:700, color:G, letterSpacing:1 }}>VOTING ENSEMBLE</div>
                      <div style={{ fontSize:9, color:G, opacity:0.6, marginTop:2 }}>75.2% directional accuracy</div>
                    </div>
                  </div>

                  {/* Report */}
                  <div style={{ background:"#fff", border:"1px solid #e0e0e0",
                    borderRadius:4, padding:"18px 22px", minWidth:0, overflow:"hidden" }}>
                    <div style={{ display:"flex", alignItems:"center", gap:10, marginBottom:16 }}>
                      <div style={{ width:4, height:18, background:G,
                        borderRadius:2, flexShrink:0 }} />
                      <span style={{ fontSize:13, fontWeight:700, color:"#1a1a1a" }}>
                        Executive Intelligence Report
                      </span>
                      <span style={{ fontSize:9, fontWeight:700, color:"#fff",
                        background:"#000", padding:"3px 9px", borderRadius:2,
                        letterSpacing:1, whiteSpace:"nowrap" }}>CLAUDE AI</span>
                    </div>

                    {/* Report sections — vertical layout */}
                    <div style={{ borderLeft:`3px solid ${G}`, paddingLeft:14, marginBottom:14 }}>
                      <div style={{ fontSize:10, fontWeight:700, color:G,
                        letterSpacing:1.5, marginBottom:7 }}>CORE RISKS</div>
                      <div style={{ fontSize:13, color:"#333", lineHeight:1.75,
                        whiteSpace:"normal", wordWrap:"break-word" }}>
                        {result.ai_report.split("\n")[0] || result.ai_report}
                      </div>
                    </div>

                    <div style={{ borderLeft:`3px solid ${G}`, paddingLeft:14, marginBottom:14 }}>
                      <div style={{ fontSize:10, fontWeight:700, color:G,
                        letterSpacing:1.5, marginBottom:7 }}>RECOMMENDATIONS</div>
                      <div style={{ fontSize:13, color:"#333", lineHeight:1.75,
                        whiteSpace:"pre-wrap", wordWrap:"break-word" }}>
                        {result.ai_report}
                      </div>
                    </div>

                    <div style={{ background:"#f9f9f9", border:"1px solid #e8e8e8",
                      borderLeft:"3px solid #1a1a1a", borderRadius:3, padding:"13px 15px" }}>
                      <div style={{ fontSize:10, fontWeight:700, color:"#1a1a1a",
                        letterSpacing:1.5, marginBottom:6 }}>OPTIMAL TIMING</div>
                      <div style={{ fontSize:13, color:"#333", lineHeight:1.75,
                        whiteSpace:"normal", wordWrap:"break-word" }}>
                        Delay full deployment until market conditions stabilize.
                        Monitor VIX and consumer confidence indicators before scaling.
                      </div>
                    </div>
                  </div>
                </div>

                {/* Footer */}
                <div style={{ marginTop:12, background:"#fff", border:"1px solid #e0e0e0",
                  borderRadius:4, padding:"9px 14px", display:"flex",
                  justifyContent:"space-between", alignItems:"center" }}>
                  <div style={{ display:"flex", alignItems:"center", gap:8 }}>
                    <div style={{ width:6, height:6, borderRadius:"50%",
                      background:G, flexShrink:0 }} />
                    <span style={{ fontSize:10, color:"#999" }}>
                      Analysis completed · {new Date().toLocaleString("en-US",{hour12:false})} · Model v2.1.0
                    </span>
                  </div>
                  <div style={{ display:"flex", border:"1px solid #e0e0e0",
                    borderRadius:3, overflow:"hidden", flexShrink:0 }}>
                    {[["S3","Storage"],["Glue","ETL"],["SageMaker","ML"],["Bedrock","GenAI"],["Athena","Query"]].map(([s,l],i,arr)=>(
                      <div key={s} style={{ padding:"4px 11px", textAlign:"center",
                        borderRight: i<arr.length-1 ? "1px solid #e0e0e0" : "none" }}>
                        <div style={{ fontSize:9, fontWeight:700, color:G }}>{s}</div>
                        <div style={{ fontSize:8, color:"#999" }}>{l}</div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </>
  );
}
