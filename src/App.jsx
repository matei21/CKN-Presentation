import { useState, useEffect } from 'react'
import axios from 'axios'

// Componentele tale (Person C)
import { FilterSphere } from './components/FilterSphere';
import { GaussianTarget } from './components/GaussianTarget';
import { PolynomialArcs } from './components/PolynomialArcs';
import InfoPage from './InfoPage';

export default function App() {
  // --- STATE ---
  const [dataset, setDataset] = useState('DeepM6A');
  const [kernel, setKernel] = useState('CKN-Polynomial');
  const [inputText, setInputText] = useState('');
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [activeIndex, setActiveIndex] = useState(0);
  const [view, setView] = useState('analysis'); // 'analysis' sau 'info'

  const kernels = ['Standard CNN', 'CKN-Linear', 'CKN-Polynomial', 'CKN-Spherical', 'CKN-Gaussian'];

  // --- CONFIGURAȚIE ---
  const styles = {
    'MNIST':    { color: '#3b82f6', secondary: '#1e40af', label: 'Cifre Hand-written', type: 'image' },
    'CIFAR-10': { color: '#10b981', secondary: '#065f46', label: 'Obiecte/Imagini', type: 'image' },
    'Ecoli':    { color: '#f59e0b', secondary: '#92400e', label: 'DNA Promoter', type: 'bio', allowed: 'ACGT' },
    'ALKBH5':   { color: '#ec4899', secondary: '#9d174d', label: 'RNA Binding', type: 'bio', allowed: 'ACGU' },
    'PTBv1':    { color: '#8b5cf6', secondary: '#4c1d95', label: 'RNA Splicing', type: 'bio', allowed: 'ACGU' },
    'DeepM6A':  { color: '#ec4899', secondary: '#9d174d', label: 'DeepM6A ARN', type: 'bio', allowed: 'ACGU' },
  };

  // 1. Definim datele extrase din scripturile tale .py pentru tabele
const benchmarkData = {
  'MNIST': [
    { model: 'CNN', acc: '0.991', auc: '1.000', train: '4242 ms', infer: '526 ms' },
    { model: 'Linear CKN', acc: '0.991', auc: '1.000', train: '4283 ms', infer: '262 ms' }, //
    { model: 'Polynomial CKN', acc: '0.992', auc: '1.000', train: '4613 ms', infer: '1109 ms' }
  ],
  'CIFAR-10': [
    { model: 'CNN', acc: '72.45%', auc: 'N/A', train: '3.12s', infer: '45ms' }, //
    { model: 'Linear CKN', acc: '68.12%', auc: 'N/A', train: '2.95s', infer: '22ms' },
    { model: 'Spherical CKN', acc: '70.33%', auc: 'N/A', train: '4.50s', infer: '38ms' }
  ]
};

// 2. Componenta pentru Tabelul de Benchmark
const BenchmarkTable = ({ data, datasetName, themeColor }) => (
  <div className="glass-panel" style={{ height: 'auto', flex: 2 }}>
    <h3 style={{ color: themeColor, fontSize: '12px', letterSpacing: '2px' }} className="mb-4 uppercase">
      {datasetName} - BENCHMARK DATA
    </h3>
    <table className="w-full text-left text-xs border-collapse">
      <thead>
        <tr style={{ borderBottom: `1px solid ${themeColor}33`, color: themeColor }}>
          <th className="pb-2">Model</th>
          <th className="pb-2">Mean Acc</th>
          <th className="pb-2">Mean AUC</th>
          <th className="pb-2">Train Time</th>
          <th className="pb-2">Infer Time</th>
        </tr>
      </thead>
      <tbody className="text-slate-300">
        {data.map((row, i) => (
          <tr key={i} style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
            <td className="py-3 font-bold" style={{ color: themeColor }}>{row.model}</td>
            <td className="py-3">{row.acc}</td>
            <td className="py-3">{row.auc}</td>
            <td className="py-3">{row.train}</td>
            <td className="py-3">{row.infer}</td>
          </tr>
        ))}
      </tbody>
    </table>
  </div>
);

// 3. Componenta pentru Archeological Insights
const ArcheologicalInsights = ({ themeColor, dataset }) => (
  <div className="glass-panel" style={{ flex: 1 }}>
    <h3 style={{ color: themeColor, fontSize: '12px' }} className="mb-4 uppercase">Archeological Insights</h3>
    <div className="space-y-4">
      <p className="text-sm font-bold">
        Verdict: {dataset === 'MNIST' ? 'Linear CKN domină eficiența, fiind de 2x mai rapid la inferență.' : 'Spherical CKN oferă cel mai bun balans între acuratețe și latență.'}
      </p>
      <div style={{ borderLeft: `3px solid ${themeColor}`, paddingLeft: '15px', fontStyle: 'italic', fontSize: '13px', color: '#94a3b8' }}>
        "CKN-urile sunt o alternativă validă la CNN, oferind fiabilitate, precizie superioară în anumite cazuri și viteze de execuție remarcabile."
      </div>
      <div className="mt-10 text-[10px] opacity-30 text-right uppercase tracking-[0.2em]">
        [3D Weight Sphere Engine - Person C]
      </div>
    </div>
  </div>
);

  const themeColor = styles[dataset]?.color || '#ffffff';
  const secondaryColor = styles[dataset]?.secondary || '#333333';
  const isBio = styles[dataset].type === 'bio';

  

  const handleBioInputChange = (e) => {
    let val = e.target.value.toUpperCase();
    const allowedChars = styles[dataset].allowed || 'ACGTU';
    const regex = new RegExp(`[^${allowedChars}]`, 'g'); // Validare caractere permise
    val = val.replace(regex, '').slice(0, 100); // Truncare la 100 caractere
    setInputText(val);
  };

  // --- API / PREDICT ---
  const runPrediction = async () => {
    setLoading(true);
    try {
      const response = await axios.post('http://localhost:8000/predict', {
        dataset: dataset,
        model_type: kernel,
        input_data: inputText
      });
      setResults(response.data);
      setView('analysis');
    } catch (error) {
      console.error("Predict Error:", error);
      setLoading(false);
    }finally {
    // Blocul FINALLY se execută MEREU (și pe succes, și pe eroare)
    setLoading(false); // Butonul revine la "Execute CKN Analysis"
  }
};

  useEffect(() => {
    setResults(null);
    setInputText('');
  }, [dataset]);

  return (
    <div className="animated-bg" style={{ 
      display: 'flex', width: '100vw', height: '100vh', 
      fontFamily: 'monospace', color: 'white',
      backgroundImage: `linear-gradient(-45deg, #020617, ${styles[dataset].secondary}66, #020617, ${styles[dataset].color}33)`,
      backgroundSize: '400% 400%',
    }}>
      
      {/* CSS-ul TĂU RESTAURAT */}
      <style>{`
        @keyframes gradientMove {
          0% { background-position: 0% 50%; }
          50% { background-position: 100% 50%; }
          100% { background-position: 0% 50%; }
        }
        .animated-bg { animation: gradientMove 15s ease infinite; transition: all 1s ease; }
        .glass-panel {
          background: rgba(0, 0, 0, 0.4);
          backdrop-filter: blur(15px);
          border: 1px solid rgba(255, 255, 255, 0.1);
          border-radius: 16px; padding: 24px;
          display: flex; flex-direction: column; height: 100%;
        }
        .sidebar-btn {
          padding: 15px; border: none; border-radius: 8px;
          font-weight: bold; cursor: pointer; text-align: left;
          transition: 0.3s; margin-bottom: 10px;
        }
        .loading-pulse { animation: pulse 1.5s infinite; }
        @keyframes pulse { 0% { opacity: 0.5; } 50% { opacity: 1; } 100% { opacity: 0.5; } }
      `}</style>

      {/* --- SIDEBAR --- */}
      <nav className="w-64 bg-[#051118] border-r border-slate-800 p-8 flex flex-col shrink-0">
        <h2 style={{ color: themeColor }} className="font-bold tracking-widest mb-10 text-lg uppercase transition-all">ARCHEOLOGY OS</h2>
        <div className="flex-1 space-y-2">
          <button 
            onClick={() => setView('info')} 
            className={`sidebar-btn w-full mt-4 text-xs tracking-widest uppercase ${view === 'info' ? 'bg-white text-black font-bold' : 'text-slate-500'}`}
          >
             Info 
          </button>
          {Object.keys(styles).map(name => (
            <button 
              key={name}
              onClick={() => {setDataset(name); setView('analysis');}}
              className={`sidebar-btn w-full text-left px-4 py-3 rounded font-bold text-sm ${
                dataset === name && view === 'analysis' ? 'sidebar-btn-active' : 'text-slate-500 hover:bg-slate-900'
              }`}
            >
              {name}
            </button>
          ))}
        </div>

        <div className="mt-6 pt-6 border-t border-slate-800">
          <label className="text-[10px] text-slate-500 uppercase block mb-2 tracking-widest">Kernel Architecture</label>
          <select 
            value={kernel}
            onChange={(e) => setKernel(e.target.value)}
            className="w-full bg-[#0a1a24] border border-cyan-900/50 p-2 rounded text-xs text-cyan-400 outline-none"
            style={{ borderColor: `${themeColor}66`, color: themeColor }}
          >
            {kernels.map(k => <option key={k} value={k}>{k}</option>)}
          </select>
        </div>
      </nav>

      {/* --- MAIN CONTENT --- */}
      <main className="flex-1 p-10 flex flex-col gap-6 overflow-y-auto">
        {view === 'info' ? (
          /* PAGINA DE INFORMAȚII - DOAR ASTA SE AFIȘEAZĂ */
          <InfoPage themeColor={themeColor} />
        ) : (
          /* PAGINA DE ANALIZĂ (WORKBENCH) */
          <>
        {/* Header Section */}
        <header>
          <div className="flex items-baseline gap-4">
            <h1 className="text-5xl font-bold text-white uppercase tracking-tighter">{dataset}</h1>
            <span style={{ color: themeColor }} className="text-xl italic">/ {kernel}</span>
          </div>
          <p className="text-slate-400 mt-2">{styles[dataset].label}</p>
        </header>


        {/* Prediction Results Banner */}
        {results && (
          <div className="glass-panel p-6 flex justify-between items-center" style={{ borderBottom: `2px solid ${themeColor}`}}>
             <div>
                <span style={{ color: themeColor }} className="text-xs font-bold block mb-1">PROBABILITY</span>
                <span className="text-4xl font-black text-white">{(results.prediction * 100).toFixed(2)}%</span>
             </div>
             <div className="text-right">
                <span style={{ color: themeColor }} className="text-xs font-bold block mb-1">CLASSIFICATION</span>
                <span className={`text-3xl font-bold ${results.label === 'POSITIVE' ? 'text-green-400' : 'text-red-400'}`}>
                  {results.label}
                </span>
             </div>
          </div>
        )}

        {/* Input Panel */}
        {isBio ? (
          <>
        <div className="glass-panel p-6" style={{ height: 'auto' }}>
          {styles[dataset].type === 'bio' ? (
            <textarea 
              value={inputText}
              onChange={handleBioInputChange}
              placeholder={`Please introduce a sequence ${styles[dataset].allowed}... (max 100)`}
              className="w-full bg-black/40 border border-slate-800 p-4 rounded-lgfont-mono outline-none h-24 resize-none transition-all"
              style={{ borderColor: `${themeColor}44`, color: themeColor }}
            />
          ) : (
            <div className="flex gap-4">
               {[1,2,3,4,5].map(i => (
                 <button key={i} onClick={() => setInputText(`img_sample_${i}`)} className="px-6 py-2 bg-slate-800 rounded hover:bg-slate-700 transition-colors text-sm">Example {i}</button>
               ))}
            </div>
          )}
          <button 
            onClick={runPrediction} 
            disabled={loading}
            className="mt-4 px-10 py-3 font-black rounded uppercase tracking-widest disabled:opacity-50 transition-all"
            style={{ backgroundColor: themeColor, color: 'black' }}
            >
            {loading ? 'Analyzing...' : 'Execute CKN Analysis'}
          </button>
        </div>

        {/* Visualizations Grid */}
        

        <div className="grid grid-cols-12 gap-6 flex-1 min-h-[500px]">
          
          {/* Panou 1: Heatmap & Arcs */}
          <div className="col-span-7 glass-panel p-6 relative overflow-hidden">
            <h3 style={{ color: themeColor }} className="text-xs font-bold tracking-widest mb-4 uppercase">Feature Mapping & Arcs</h3>
            <div className="relative h-full w-full bg-black/20 rounded p-4 border border-slate-900/50">
               <div className="grid grid-cols-20 gap-1 h-full w-full">
                  {(results?.heatmap || Array(400).fill(0)).map((val, i) => (
                    <div 
                      key={i} 
                      className="rounded-[1px] transition-all duration-700"
                      style={{ 
                        backgroundColor: '#00bcd4', 
                        opacity: loading ? 0.1 : val 
                      }} 
                    />
                  ))}
               </div>
               {results && kernel === 'CKN-Polynomial' && (
                 <PolynomialArcs arcs={results.polynomial_arcs} />
               )}
            </div>
          </div>

          {/* Panou 2: Insights (Sphere/Bullseye) */}
          <div className="col-span-5 glass-panel p-6 flex flex-col">
            <h3 style={{ color: themeColor }} className="text-xs font-bold tracking-widest mb-4 uppercase">Model Insights</h3>
            
            <div className="mb-6 space-y-2 text-xs">
              <p className="text-slate-500 uppercase tracking-widest">Active Kernel: <span className="text-slate-200">{kernel}</span></p>
              <p className="text-slate-500">Monitoring activation stability and inference latency.</p>
            </div>

            <div className="flex-1 bg-black/30 rounded border border-dashed border-slate-800 flex items-center justify-center overflow-hidden">
              {results ? (
                kernel === 'CKN-Spherical' ? (
                  <FilterSphere pointsData={results.sphere_vectors} themeColor={themeColor} />
                ) : (
                  <GaussianTarget score={1 - results.gaussian_dist} themeColor={themeColor}/>
                )
              ) : (
                <span className="text-slate-700 text-[10px] tracking-[0.3em] uppercase">[Waiting for execution]</span>
              )}
            </div>
          </div>
        </div>
        </>
        ) : (
    // Vizualizarea stil Screenshot pentru MNIST/CIFAR
      <div className="flex gap-6 flex-1 min-h-0">
                <BenchmarkTable data={benchmarkData[dataset]} datasetName={dataset} themeColor={themeColor} />
                <ArcheologicalInsights themeColor={themeColor} dataset={dataset} />
                
              </div>
      )}
      </>
    )}
      </main>
    </div>
  )
}