import {
  Chart,
  LineController,
  LineElement,
  PointElement,
  LinearScale,
  TimeScale,
  Title,
  Tooltip,
  Legend,
  CategoryScale,
  Filler,
} from 'chart.js';

Chart.register(
  LineController,
  LineElement,
  PointElement,
  LinearScale,
  TimeScale,
  Title,
  Tooltip,
  Legend,
  CategoryScale,
  Filler,
);

export class RealTimePlot {
  chart: Chart;
  maxPoints: number;

  constructor(
    canvas: HTMLCanvasElement,
    title: string,
    labels: string[],
    colors: string[],
    maxPoints = 200,
  ) {
    this.maxPoints = maxPoints;
    const datasets = labels.map((label, i) => ({
      label,
      data: [] as number[],
      borderColor: colors[i % colors.length],
      backgroundColor: colors[i % colors.length] + '33',
      borderWidth: 1.5,
      pointRadius: 0,
      tension: 0.2,
      fill: false,
    }));

    this.chart = new Chart(canvas, {
      type: 'line',
      data: { labels: [] as string[], datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        interaction: { mode: 'index', intersect: false },
        plugins: {
          title: { display: true, text: title, color: '#e2e8f0', font: { size: 12 } },
          legend: {
            labels: { color: '#94a3b8', boxWidth: 12, font: { size: 10 } },
          },
        },
        scales: {
          x: {
            display: true,
            ticks: { color: '#64748b', maxTicksLimit: 6, font: { size: 10 } },
            grid: { color: 'rgba(255,255,255,0.06)' },
          },
          y: {
            display: true,
            ticks: { color: '#64748b', font: { size: 10 } },
            grid: { color: 'rgba(255,255,255,0.06)' },
          },
        },
      },
    });
  }

  push(values: number[], label: string) {
    const labels = this.chart.data.labels as string[];
    labels.push(label);
    if (labels.length > this.maxPoints) labels.shift();

    for (let i = 0; i < values.length; i++) {
      const ds = this.chart.data.datasets[i];
      if (!ds) continue;
      (ds.data as number[]).push(values[i]);
      if (ds.data.length > this.maxPoints) (ds.data as number[]).shift();
    }
    this.chart.update('none');
  }
}
