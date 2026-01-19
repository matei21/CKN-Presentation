import * as THREE from 'three';
import { useEffect, useRef } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Sphere, Points, PointMaterial, Environment } from '@react-three/drei';

// Componenta primește acum themeColor din App.jsx
export function FilterSphere({ pointsData, themeColor }) {
  // Fallback în cazul în care culoarea nu este definită
  const activeColor = themeColor || "#22d3ee";

  // Convertim datele dintr-un array de obiecte [{x,y,z}] într-un format plat pentru Three.js
  const positions = new Float32Array((pointsData?.length || 0) * 3);
  if (pointsData) {
    for (let i = 0; i < pointsData.length; i++) {
      positions[i * 3] = pointsData[i].x;
      positions[i * 3 + 1] = pointsData[i].y;
      positions[i * 3 + 2] = pointsData[i].z;
    }
  }

  return (
    <div className="h-[400px] w-full bg-[#0a0e17]/50 rounded-xl overflow-hidden border border-white/5 relative">
        <div 
          style={{ color: activeColor }} 
          className="absolute top-3 left-3 text-[10px] font-mono z-10 bg-black/40 px-3 py-1 rounded-full border border-white/10 tracking-[0.2em] uppercase"
        >
            RKHS Geometry View
        </div>

      <Canvas camera={{ position: [0, 0, 2.5] }}>
        <color attach="background" args={['#0a0e17']} />
        <ambientLight intensity={0.4} />
        
        {/* Lumina ambientală preia acum culoarea datasetului */}
        <pointLight position={[10, 10, 10]} intensity={1.5} color={activeColor} />
        
        <OrbitControls enableZoom={true} autoRotate autoRotateSpeed={0.5} />
        
        {/* Sfera de ghidaj: firele de sârmă (wireframe) devin subtile în culoarea temei */}
        <Sphere args={[1, 32, 32]}>
          <meshBasicMaterial color={activeColor} wireframe transparent opacity={0.05} />
        </Sphere>

        {/* Reprezentarea vectorilor învățați de kernelul sferic */}
        <Points positions={positions} stride={3}>
          <PointMaterial 
            transparent 
            color={activeColor} 
            emissive={activeColor}
            emissiveIntensity={3}
            size={0.06} 
            sizeAttenuation={true} 
            depthWrite={false} 
            toneMapped={false}
            blending={THREE.AdditiveBlending}
          />
        </Points>
        <Environment preset="night" />
      </Canvas>
    </div>
  );
}