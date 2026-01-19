import { useState, useEffect } from 'react'
import axios from 'axios'

export default function App() {
  const [dataset, setDataset] = useState('MNIST');
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);

  // Configurația stilurilor cerută de Matei
  const styles = {
    'MNIST': { color: '#3b82f6', secondary: '#1e40af', label: 'Cifre Hand-written', grid: 'repeat(28, 1fr)' },
    'CIFAR-10': { color: '#10b981', secondary: '#065f46', label: 'Obiecte/Imagini', grid: 'repeat(32, 1fr)' },
    'RBP-24': { color: '#f59e0b', secondary: '#92400e', label: 'Secvențe Proteine', grid: 'repeat(50, 1fr)' },
    'm6': { color: '#ec4899', secondary: '#9d174d', label: 'Dataset m6', grid: 'repeat(20, 1fr)' }
  };

  // Funcția care va lua date de la Persoana A (Backend)
  const fetchData = async (selectedDataset) => {
    setLoading(true);
    try {
      // ATENȚIE: Aici vei pune URL-ul colegului tău când e gata backend-ul
      // const response = await axios.get(`http://localhost:8000/predict?ds=${selectedDataset}`);
      // setData(response.data);
      
      // Momentan simulăm un delay de rețea și date random
      setTimeout(() => {
        const fakeData = Array.from({ length: 400 }, () => Math.random());
        setData(fakeData);
        setLoading(false);
      }, 800);
      
    } catch (error) {
      console.error("Eroare la preluarea datelor:", error);
      setLoading(false);
    }
  };

  // Rulează funcția de fetch ori de câte ori se schimbă dataset-ul
  useEffect(() => {
    fetchData(dataset);
  }, [dataset]);

  return (
    <div className="animated-bg" style={{ 
      display: 'flex', width: '100vw', height: '100vh', 
      fontFamily: 'monospace', color: 'white',
      backgroundImage: `linear-gradient(-45deg, #020617, ${styles[dataset].secondary}, #020617, ${styles[dataset].color})`,
      backgroundSize: '400% 400%',
    }}>
      
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

      {/* SIDEBAR */}
      <nav style={{ 
        width: '280px', padding: '30px', background: 'rgba(0,0,0,0.6)', 
        borderRight: '1px solid rgba(255,255,255,0.1)', display: 'flex', flexDirection: 'column'
      }}>
        <h2 style={{ letterSpacing: '3px', color: styles[dataset].color, marginBottom: '40px' }}>ARCHEOLOGY OS</h2>
        {Object.keys(styles).map(name => (
          <button 
            key={name}
            onClick={() => setDataset(name)}
            className="sidebar-btn"
            style={{
              background: dataset === name ? styles[name].color : 'rgba(255,255,255,0.05)',
              color: dataset === name ? 'black' : 'white',
            }}
          >
            {name}
          </button>
        ))}
        
        <div style={{ marginTop: 'auto', fontSize: '0.7rem', opacity: 0.5 }}>
          v1.0.4 | Status: {loading ? 'Computing...' : 'Ready'}
        </div>
      </nav>

      {/* MAIN CONTENT */}
      <main style={{ flex: 1, padding: '50px', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        <header style={{ marginBottom: '40px' }}>
          <h1 style={{ fontSize: '4rem', margin: 0, textShadow: `0 0 30px ${styles[dataset].color}` }}>{dataset}</h1>
          <p style={{ opacity: 0.8, fontSize: '1.2rem' }}>{styles[dataset].label}</p>
        </header>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '30px', flex: 1, minHeight: 0 }}>
          
          {/* PANOU HEATMAP (Partea ta) */}
          <div className="glass-panel">
            <h3 style={{ color: styles[dataset].color, marginTop: 0 }}>2D ACTIVATION MAP</h3>
            <div style={{ 
              flex: 1, marginTop: '15px', overflowY: 'auto',
              display: 'grid', gridTemplateColumns: styles[dataset].grid, gap: '2px',
              padding: '10px', background: 'rgba(0,0,0,0.2)', borderRadius: '8px'
            }}>
              {loading ? (
                <div className="loading-pulse" style={{ gridColumn: '1/-1', textAlign: 'center', paddingTop: '100px' }}>
                  Processing Kernels...
                </div>
              ) : (
                data?.map((val, i) => (
                  <div key={i} style={{ 
                    aspectRatio: '1/1', backgroundColor: styles[dataset].color, 
                    opacity: val, borderRadius: '1px' 
                  }} />
                ))
              )}
            </div>
          </div>

          {/* PANOU SFERĂ 3D (Partea Colegului C) */}
          <div className="glass-panel">
            <h3 style={{ color: styles[dataset].color, marginTop: 0 }}>3D WEIGHT SPHERE</h3>
            <div style={{ 
              flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', 
              border: '1px dashed rgba(255,255,255,0.1)', borderRadius: '8px', marginTop: '15px' 
            }}>
               <div style={{ textAlign: 'center', opacity: 0.5 }}>
                  <p>3D Engine Offline</p>
                  <small>Integrate Person C Component Here</small>
               </div>
            </div>
          </div>

        </div>
      </main>
    </div>
  )
}