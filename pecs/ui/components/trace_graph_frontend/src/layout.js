export function openLayoutOptions(animate = true) {
  return {
    name: "cose",
    animate,
    animationDuration: 650,
    randomize: true,
    fit: true,
    padding: 70,
    nodeRepulsion: 12000,
    idealEdgeLength: 150,
    edgeElasticity: 80,
    nestingFactor: 1.2,
    gravity: 0.15,
    numIter: 1200,
    componentSpacing: 140,
    nodeOverlap: 24,
  };
}
