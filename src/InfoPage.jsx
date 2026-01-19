import React, { useState } from 'react';

// --- FULL DATASET FROM YOUR RESEARCH DOCUMENT ---
const performanceData = {
  MNIST: {
    metrics: ['Accuracy', 'Inference Speed'],
    cnn: [0.991, 526], poly: [0.992, 1109], linear: [0.991, 262],
    verdict: "Linear CKN offers a 50% latency reduction while matching CNN accuracy."
  },
  CIFAR10: {
    metrics: ['Accuracy', 'Inference Speed'],
    cnn: [0.554, 114], poly: [0.624, 90], linear: [0.570, 47],
    verdict: "Spherical CKN leads in accuracy, but Polynomial is the best 'value' choice."
  },
  Ecoli: {
    metrics: ['Accuracy', 'Inference Speed'],
    cnn: [0.882, 56.5], poly: [0.901, 72.5], linear: [0.891, 61.8],
    verdict: "Polynomial CKN is the clear winner for promoter recognition (0.965 AUC)."
  },
  ALKBH5: {
    metrics: ['Accuracy', 'Inference Speed'],
    cnn: [0.625, 124], poly: [0.635, 132], linear: [0.631, 122],
    verdict: "Polynomial CKN captures the best non-linear binding interactions."
  },
  PTBv1: {
    metrics: ['Accuracy', 'Inference Speed'],
    cnn: [0.989, 1661], poly: [0.988, 1155], linear: [0.988, 1058],
    verdict: "Linear CKN is 35% faster than CNN at inference in high-throughput splicing."
  },
  m6A: {
    metrics: ['Accuracy', 'Inference Speed'],
    cnn: [0.747, 296], poly: [0.748, 297], linear: [0.746, 221],
    verdict: "Polynomial CKN offers the best stability (lowest Std AUC) for m6A detection."
  }
};

const InfoSection = ({ title, children, color }) => (
  <section className="mb-12">
    <h2 style={{ color }} className="text-2xl font-bold mb-6 tracking-widest uppercase border-b border-white/10 pb-2">
      {title}
    </h2>
    <div className="text-slate-300 leading-relaxed space-y-4">
      {children}
    </div>
  </section>
);

export default function InfoPage({ themeColor }) {
  const [selectedDataset, setSelectedDataset] = useState('MNIST');
  const [metricIndex, setMetricIndex] = useState(0); // 0 = Acc, 1 = Speed

  const data = performanceData[selectedDataset];
  const maxVal = Math.max(data.cnn[metricIndex], data.poly[metricIndex], data.linear[metricIndex]);
  
  // Chart Logic: Accuracy (Higher is better) vs Latency (Lower is better)
  const getWidth = (val) => {
    if (metricIndex === 0) return (val * 100) + '%';
    return (100 - (val / maxVal * 100) + 25) + '%';
  };

  return (
    <div className="p-10 max-w-6xl mx-auto overflow-y-auto h-full space-y-16">
      <header className="text-center">
        <h1 className="text-6xl font-black text-white mb-4 tracking-tighter uppercase">Archeology OS</h1>
        <p className="text-xl text-slate-400 italic">Complete Comparative Benchmark Suite</p>
      </header>

      <p className="text-sm text-slate-400">Our project investigates the performance and theoretical foundations of Convolutional Kernel Networks (CKN) compared to standard Convolutional Neural Networks (CNN). While CNNs are the standard for image recognition, CKNs offer a unique perspective by leveraging the kernel trick within a Reproducing Kernel Hilbert Space (RKHS). We aim to determine if CKNs can create stable, robust representations similar to or better than CNNs across varying datasets: MNIST, CIFAR-10, alongside bioinformatics datasets. We evaluate both approaches based on accuracy, robustness to transformation, and resource efficiency.
</p>


<InfoSection title="1. Research Abstract & Theoretical Framework" color={themeColor}>
        <p>
          This project presents a deep-dive comparative study into <strong>Convolutional Kernel Networks (CKNs)</strong>. CKNs represent a paradigm shift in deep learning, 
          bridging the gap between the hierarchical feature extraction of standard <strong>CNNs</strong> and the mathematical rigor of <strong>Kernel Methods</strong>.
        </p>
        
        <p>
          By implementing the "kernel trick" within a <strong>Reproducing Kernel Hilbert Space (RKHS)</strong>, these models map local image or sequence patches to a 
          high-dimensional space where linear separators become significantly more powerful. We evaluate four kernel types—<strong>Linear, Polynomial, Spherical, 
          and Gaussian</strong>—against a baseline CNN to assess their stability, accuracy, and computational "cost-of-discovery."
        </p>
      </InfoSection>


      {/* --- THE INTERACTIVE DISCOVERY ENGINE --- */}
      <InfoSection title="2. Comparative Performance Discovery" color={themeColor}>
        <div className="glass-panel p-8">
          <div className="flex flex-wrap gap-2 mb-8">
            {Object.keys(performanceData).map(ds => (
              <button key={ds} onClick={() => setSelectedDataset(ds)} 
                className={`px-6 py-2 text-[10px] font-black uppercase tracking-widest transition-all rounded-sm ${selectedDataset === ds ? 'bg-white text-black' : 'bg-white/5 text-slate-500 hover:bg-white/10'}`}>
                {ds}
              </button>
            ))}
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-12 items-center">
            {/* Chart Column */}
            <div className="lg:col-span-2 space-y-8">
              <div className="flex justify-between items-end">
                <h3 className="text-white font-bold uppercase tracking-widest text-sm">
                  {metricIndex === 0 ? "Classification Accuracy (%)" : "Inference Latency (ms)"}
                </h3>
                <button onClick={() => setMetricIndex(metricIndex === 0 ? 1 : 0)} className="text-[10px] text-slate-500 underline uppercase tracking-widest">
                  Switch Metric
                </button>
              </div>

              {/* BARS */}
              {[
                { label: 'Standard CNN', val: data.cnn[metricIndex], color: '#64748b' },
                { label: 'Polynomial CKN', val: data.poly[metricIndex], color: themeColor },
                { label: 'Linear CKN', val: data.linear[metricIndex], color: '#f8fafc' }
              ].map(bar => (
                <div key={bar.label} className="space-y-2">
                  <div className="flex justify-between text-[10px] uppercase font-bold tracking-widest">
                    <span style={{ color: bar.label.includes('Poly') ? themeColor : 'inherit' }}>{bar.label}</span>
                    <span>{metricIndex === 0 ? (bar.val * 100).toFixed(1) + '%' : bar.val + 'ms'}</span>
                  </div>
                  <div className="h-1 w-full bg-white/5 rounded-full overflow-hidden">
                    <div className="h-full transition-all duration-1000 ease-out" style={{ width: getWidth(bar.val), backgroundColor: bar.color }} />
                  </div>
                </div>
              ))}
            </div>

            {/* Insight Column */}
            <div className="bg-white/5 p-6 border-l-2" style={{ borderColor: themeColor }}>
              <h4 className="text-[10px] uppercase font-bold mb-4 tracking-[0.3em]" style={{ color: themeColor }}>Archeological Verdict</h4>
              <p className="text-sm italic text-slate-300 leading-relaxed">"{data.verdict}"</p>
              
            </div>
          </div>
        </div>
      </InfoSection>

      {/* --- DETAILED RESEARCH DATA --- */}
      <InfoSection title="3. Full Study Context" color={themeColor}>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-12">
          <div>
            <h3 className="text-white font-bold mb-4 uppercase text-sm tracking-widest">Bioinformatics Impact</h3>
            <p className="text-sm text-slate-400">
              In molecular biology, accurately classifying DNA/RNA sequences (Promoters, m6A, ALKBH5) is a non-linear pattern recognition task. 
              The <strong>Polynomial CKN</strong> consistently provides the highest AUC and stability, outperforming standard convolutions by capturing complex geometric motifs.
            </p>
            
          </div>
          <div>
            <h3 className="text-white font-bold mb-4 uppercase text-sm tracking-widest">Computational Efficiency</h3>
            <p className="text-sm text-slate-400">
              The <strong>Linear CKN</strong> represents the best blend of speed and performance. On datasets like PTBv1, it is <strong>35% faster</strong> 
              than CNNs, allowing scientists to run thousands of additional simulations in the same timeframe.
            </p>
            
          </div>
        </div>
      </InfoSection>

      
     <InfoSection title="4. The Bio-Pattern Recognition Frontier" color={themeColor}>
        <div className="space-y-12">
          <div className="grid md:grid-cols-2 gap-8">
            <div className="space-y-4">
              <h3 className="text-white font-bold uppercase text-sm tracking-widest">Promoter Site Discovery (E-coli)</h3>
              <p className="text-sm text-slate-400">
                In molecular biology, a <strong>promoter</strong> is a DNA region that initiates transcription. Accurate classification of these sites is a fundamental 
                bioinformatics challenge due to the non-linear motifs within sequences.
              </p>
              
              <p className="text-sm text-slate-400">
                Our study found that the <strong>Polynomial CKN</strong> is the superior discriminator, achieving a Mean AUC of <strong>0.965</strong>. It provides 
                the most consistent performance for accurately identifying functional gene-start signals.
              </p>
            </div>
            <div className="space-y-4">
              <h3 className="text-white font-bold uppercase text-sm tracking-widest">ALKBH5 & Cancer Association</h3>
              <p className="text-sm text-slate-400">
                ALKBH5 is an RNA demethylase. RNA molecules that <strong>do not bind</strong> to ALKBH5 preserve their methylation status, which is crucial for 
                RNA stability. Identifying binding patterns is critical for understanding cancer pathways.
              </p>
              <p className="text-sm text-slate-400">
                The <strong>Polynomial CKN</strong> achieved a 63.5% accuracy, proving its ability to capture complex non-linear interactions within the RNA data 
                that traditional convolutions often misidentify.
              </p>
            </div>
          </div>

          <div className="p-8 border border-white/5 bg-white/[0.01]">
            <h3 className="text-white font-bold uppercase text-sm tracking-widest mb-4">m6A (Methyladenosine) Binding</h3>
            <p className="text-sm text-slate-400 mb-6">
              N6-methyladenosine is the most common internal modification of mRNA. Our study on 10 independent runs showed that 
              <strong> Polynomial, Spherical, and Gaussian CKNs</strong> offer the highest stability (Lowest Std AUC: 0.0023). 
              For maximum reliability in detection scenarios, the Polynomial variant remains the top academic choice.
            </p>
            
          </div>
        </div>
      </InfoSection>

      {/* --- SECTION 4: COMPUTER VISION --- */}
      <InfoSection title="5. Computer Vision Efficiency Benchmarks" color={themeColor}>
        <div className="grid md:grid-cols-2 gap-12">
          <div className="space-y-4">
            <h3 className="text-white font-bold uppercase text-sm tracking-widest">MNIST Stability Analysis</h3>
            <p className="text-sm text-slate-400">
              Tested over 100 runs, CKNs (Linear and Polynomial) matched or exceeded CNN performance. The <strong>Linear CKN</strong> dominated 
              inference efficiency, showing a <strong>2x speed advantage</strong> over the CNN baseline.
            </p>
            
          </div>
          <div className="space-y-4">
            <h3 className="text-white font-bold uppercase text-sm tracking-widest">CIFAR-10 Depth Comparison</h3>
            <p className="text-sm text-slate-400">
              The <strong>Spherical CKN</strong> achieved 63.45% accuracy. A sub-study compared a SOTA CNN (93.44%) with our best CKN. 
              The CKN reached <strong>92.75%</strong>, proving that the geometric robustness of kernel methods remains competitive even 
              in deep-layer architectures, within a narrow 1% margin.
            </p>
          </div>
        </div>
      </InfoSection>

      {/* --- SECTION 5: THE FINAL VERDICT --- */}
      <footer className="mt-40 border-t-4 border-white pt-16" style={{ borderColor: themeColor }}>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-20">
          <div>
            <h2 className="text-4xl font-black text-white uppercase tracking-tighter mb-6">The Archeological Verdict</h2>
            <div className="space-y-6 text-slate-400 italic text-lg leading-relaxed">
              <p>The transition from standard CNNs to Convolutional Kernel Networks offers a high-impact trade-off between mathematical interpretability and computational speed.</p>
              <p>For high-discovery accuracy in bioinformatics, <strong>Polynomial CKN</strong> is the gold standard. For deployment efficiency, <strong>Linear CKN</strong> is a game-changer, offering up to 35% more calculations per timeframe.</p>
            </div>
          </div>
          <div className="bg-white/5 p-10 space-y-8">
            <div>
              <h4 className="text-[10px] uppercase font-bold text-slate-500 mb-2">Computational Overhead</h4>
              <p className="text-sm text-white">Non-linear kernels (Polynomial/Spherical) require significantly more training time—up to 7x more than standard CNNs—as the price for geometric precision.</p>
            </div>
            
          </div>
        </div>
        
      </footer>
    </div>
  );
}