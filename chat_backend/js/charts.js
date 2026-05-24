/**
 * AlgoChat — Chart Renderer
 * Renders Chart.js charts in both tool cards and preview panel
 */

const ChartRenderer = {
  instances: {},

  chartColors: [
    'rgba(125, 158, 141, 0.8)',  // teal
    'rgba(142, 78, 38, 0.8)',    // orange
    'rgba(153, 45, 30, 0.7)',    // red
    'rgba(160, 185, 170, 0.8)',  // teal-light
    'rgba(175, 120, 80, 0.7)',   // orange-light
    'rgba(190, 90, 70, 0.6)',    // red-light
  ],

  chartBorders: [
    'rgb(125, 158, 141)',
    'rgb(142, 78, 38)',
    'rgb(153, 45, 30)',
    'rgb(160, 185, 170)',
    'rgb(175, 120, 80)',
    'rgb(190, 90, 70)',
  ],

  baseOptions: {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        labels: {
          color: 'rgb(100, 80, 60)',
          font: { family: "'Segoe UI', sans-serif", size: 12 },
          padding: 12,
          usePointStyle: true,
        }
      },
      tooltip: {
        backgroundColor: 'rgba(245, 237, 218, 0.95)',
        titleColor: 'rgb(60, 40, 25)',
        bodyColor: 'rgb(100, 80, 60)',
        borderColor: 'rgba(125, 158, 141, 0.3)',
        borderWidth: 1,
        padding: 10,
        cornerRadius: 8,
        titleFont: { weight: 'bold' },
      }
    },
    scales: {
      x: {
        ticks: { color: 'rgb(150, 135, 115)', font: { size: 11 } },
        grid: { color: 'rgba(200, 192, 172, 0.3)' },
      },
      y: {
        ticks: { color: 'rgb(150, 135, 115)', font: { size: 11 } },
        grid: { color: 'rgba(200, 192, 172, 0.3)' },
      }
    }
  },

  render(container, chartData, height) {
    const canvas = document.createElement('canvas');
    const wrapper = document.createElement('div');
    wrapper.className = 'chart-container';
    if (height) wrapper.style.height = height + 'px';
    wrapper.appendChild(canvas);
    container.appendChild(wrapper);

    const ctx = canvas.getContext('2d');
    const config = this.buildConfig(chartData);
    const instance = new Chart(ctx, config);
    const id = 'chart_' + Date.now() + '_' + Math.random().toString(36).substr(2, 5);
    this.instances[id] = instance;
    canvas.dataset.chartId = id;
    return id;
  },

  renderInPreview(previewContent, chartData) {
    previewContent.innerHTML = '';
    const container = document.createElement('div');
    container.className = 'preview-chart-container';
    container.style.height = '350px';
    this.render(container, chartData, 350);
    previewContent.appendChild(container);
  },

  buildConfig(chartData) {
    const type = chartData.chartType || 'bar';
    const data = this.buildData(chartData);
    const options = JSON.parse(JSON.stringify(this.baseOptions));

    if (type === 'pie' || type === 'doughnut') {
      delete options.scales;
    }

    if (type === 'scatter') {
      options.scales.x.title = { display: true, text: 'X', color: 'rgb(150,135,115)' };
      options.scales.y.title = { display: true, text: 'Y', color: 'rgb(150,135,115)' };
    }

    if (type === 'radar') {
      delete options.scales;
      options.scales = {
        r: {
          ticks: { color: 'rgb(150,135,115)', backdropColor: 'transparent' },
          grid: { color: 'rgba(200,192,172,0.3)' },
          pointLabels: { color: 'rgb(100,80,60)', font: { size: 11 } },
        }
      };
    }

    return { type, data, options };
  },

  buildData(chartData) {
    const type = chartData.chartType || 'bar';
    const source = chartData.data || {};

    if (type === 'scatter') {
      return {
        datasets: (source.datasets || []).map((ds, i) => ({
          label: ds.label || `数据集 ${i+1}`,
          data: ds.data || [],
          backgroundColor: this.chartColors[i % this.chartColors.length],
          borderColor: this.chartBorders[i % this.chartBorders.length],
          pointRadius: 4,
          pointHoverRadius: 6,
        }))
      };
    }

    const labels = source.labels || [];
    const datasets = (source.datasets || []).map((ds, i) => {
      const base = {
        label: ds.label || `数据集 ${i+1}`,
        data: ds.data || [],
        backgroundColor: (type === 'pie' || type === 'doughnut')
          ? this.chartColors
          : this.chartColors[i % this.chartColors.length],
        borderColor: (type === 'pie' || type === 'doughnut')
          ? this.chartBorders
          : this.chartBorders[i % this.chartBorders.length],
        borderWidth: type === 'pie' || type === 'doughnut' ? 2 : 2,
        tension: 0.3,
        fill: type === 'line' ? false : undefined,
        pointBackgroundColor: this.chartBorders[i % this.chartBorders.length],
        pointRadius: type === 'line' ? 3 : undefined,
      };

      if (type === 'bar') {
        base.borderRadius = 4;
        base.borderSkipped = false;
      }

      return base;
    });

    return { labels, datasets };
  },

  destroy(id) {
    if (this.instances[id]) {
      this.instances[id].destroy();
      delete this.instances[id];
    }
  },

  destroyAll() {
    Object.keys(this.instances).forEach(id => this.destroy(id));
  },

  downloadAsPNG(chartId, filename) {
    const instance = this.instances[chartId];
    if (!instance) return;
    const url = instance.toBase64Image();
    const a = document.createElement('a');
    a.href = url;
    a.download = filename || 'chart.png';
    a.click();
  },
};