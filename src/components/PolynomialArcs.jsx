export function PolynomialArcs({ arcs }) {
  // Presupunem un grid de 20 coloane (ca în exemplul de mai sus)
  const getCoords = (index) => {
    const col = index % 20;
    const row = Math.floor(index / 20);
    return { x: col * 5 + 2.5, y: row * 5 + 2.5 }; // Procente pentru SVG
  };

  return (
    <svg className="absolute inset-0 w-full h-full pointer-events-none" viewBox="0 0 100 100" preserveAspectRatio="none">
      {arcs.map((arc, i) => {
        const start = getCoords(arc.start);
        const end = getCoords(arc.end);
        
        // Calculăm un punct de control pentru curbură
        const midX = (start.x + end.x) / 2;
        const midY = (start.y + end.y) / 2 - 15; // "Ridică" curba

        return (
          <path
            key={i}
            d={`M ${start.x} ${start.y} Q ${midX} ${midY} ${end.x} ${end.y}`}
            fill="none"
            stroke="#00bcd4"
            strokeWidth={arc.strength * 0.5}
            strokeOpacity={arc.strength}
            className="animate-pulse"
          />
        );
      })}
    </svg>
  );
}