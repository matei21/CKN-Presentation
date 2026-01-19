// src/components/BioSequence.jsx
import { GitCommit } from 'lucide-react';

// Helper pentru culorile standard din bioinformatică
const getNucleotideColor = (char) => {
  switch (char.toUpperCase()) {
    case 'A': return 'text-green-400'; // Adenine
    case 'T': return 'text-red-400';   // Thymine
    case 'C': return 'text-blue-400';  // Cytosine
    case 'G': return 'text-yellow-400';// Guanine
    default: return 'text-slate-400';
  }
};

// Primește secvența (string) și un array de greutăți (numere 0-1)
export function BioSequence({ sequence, weights }) {
  return (
    <div className="p-4 bg-[#0a0e17] border border-cyan-900/50 rounded-xl shadow-[0_0_15px_rgba(34,211,238,0.1)] overflow-x-auto">
      <div className="flex items-center gap-2 mb-3">
        <GitCommit className="w-4 h-4 text-cyan-400" />
        <h3 className="text-cyan-400 font-mono text-xs uppercase tracking-wider">Input Sequence & Attention Heatmap</h3>
      </div>
      
      {/* Containerul pentru secvență, cu scroll orizontal dacă e lungă */}
      <div className="flex gap-1 min-w-max pb-2">
        {sequence.split('').map((char, index) => {
          const weight = weights[index] || 0;
          return (
            <div key={index} className="flex flex-col items-center group relative">
              
              {/* Litera nucleotidei */}
              <span className={`font-mono font-bold text-lg ${getNucleotideColor(char)} transition-all group-hover:scale-125`}>
                {char}
              </span>
              
              {/* Bara de heatmap dedesubt */}
              <div 
                className="w-5 mt-1 rounded-sm transition-all duration-300 bg-cyan-500/80 group-hover:bg-cyan-400"
                // Înălțimea și opacitatea depind de greutate
                style={{ 
                  height: `${Math.max(4, weight * 24)}px`, 
                  opacity: Math.max(0.2, weight) 
                }} 
              />
              
               {/* Tooltip la hover (opțional, dar arată pro) */}
              <div className="absolute bottom-full mb-2 hidden group-hover:block bg-slate-800 text-[10px] p-1 rounded border border-slate-600 whitespace-nowrap z-50">
                Pos: {index} | W: {weight.toFixed(2)}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}