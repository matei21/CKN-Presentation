// src/components/GaussianTarget.jsx

// Componenta primește un "score" între 0.0 și 1.0
export function GaussianTarget({ score }) {
    const svgSize = 200;
    const center = svgSize / 2;
    const maxRadius = 80;
  
    // Calculăm cât de departe de centru este punctul.
    // Scor 1 -> distanță 0. Scor 0 -> distanță maxRadius.
    const distanceFromCenter = (1 - score) * maxRadius;
  
    return (
      <div className="flex flex-col items-center p-4 bg-[#0a0e17] border border-cyan-900/50 rounded-xl shadow-[0_0_15px_rgba(34,211,238,0.1)]">
        <h3 className="text-cyan-400 font-mono text-xs mb-2 uppercase tracking-wider">Kernel Activation Proximity</h3>
        
        <svg width={svgSize} height={svgSize} viewBox={`0 0 ${svgSize} ${svgSize}`}>
          {/* Cercurile concentrice de fundal */}
          <circle cx={center} cy={center} r={maxRadius} fill="none" stroke="#1e293b" strokeWidth="1" />
          <circle cx={center} cy={center} r={maxRadius * 0.75} fill="none" stroke="#1e293b" strokeWidth="1" strokeDasharray="4 4" />
          <circle cx={center} cy={center} r={maxRadius * 0.5} fill="none" stroke="#1e293b" strokeWidth="1" />
          <circle cx={center} cy={center} r={maxRadius * 0.25} fill="none" stroke="#334155" strokeWidth="1" />
          
          {/* Centrul "ideal" */}
          <circle cx={center} cy={center} r={3} fill="#22d3ee" opacity={0.5} />
  
          {/* Linia indicatoare de la centru la punct */}
          <line 
            x1={center} y1={center} 
            x2={center + distanceFromCenter} y2={center} 
            stroke="#22d3ee" strokeWidth="2" strokeOpacity="0.4" 
          />
  
          {/* Punctul care se mișcă (Activarea curentă) */}
          <circle 
            cx={center + distanceFromCenter} 
            cy={center} 
            r={6} 
            fill="#22d3ee" 
            className="transition-all duration-500 ease-out shadow-[0_0_10px_#22d3ee]"
            style={{ filter: 'drop-shadow(0 0 4px #22d3ee)' }}
          />
        </svg>
        
        {/* Afișarea scorului numeric */}
        <div className="mt-2 font-mono">
            <span className="text-slate-400 text-xs">Match Score: </span>
            <span className={`font-bold ${score > 0.8 ? 'text-green-400' : 'text-cyan-400'}`}>
                {(score * 100).toFixed(1)}%
            </span>
        </div>
      </div>
    );
  }