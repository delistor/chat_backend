/**
 * AlgoChat — Table Renderer & Exporter
 * Renders tables in tool cards + preview panel + CSV/XLSX download
 */

const TableRenderer = {
  render(container, tableData, options = {}) {
    const { columns, rows } = tableData;
    const maxHeight = options.maxHeight || 350;
    const sortable = options.sortable !== false;

    const wrapper = document.createElement('div');
    wrapper.className = 'table-container';
    wrapper.style.maxHeight = maxHeight + 'px';

    const table = document.createElement('table');
    const thead = document.createElement('thead');
    const tbody = document.createElement('tbody');

    // Header
    const headerRow = document.createElement('tr');
    columns.forEach((col, ci) => {
      const th = document.createElement('th');
      th.textContent = col;
      if (sortable) {
        const indicator = document.createElement('span');
        indicator.className = 'sort-indicator';
        indicator.textContent = '↕';
        th.appendChild(indicator);
        th.addEventListener('click', () => this.sortTable(tbody, ci, indicator));
      }
      headerRow.appendChild(th);
    });
    thead.appendChild(headerRow);
    table.appendChild(thead);

    // Body
    rows.forEach(row => {
      const tr = document.createElement('tr');
      row.forEach(cell => {
        const td = document.createElement('td');
        td.textContent = cell;
        tr.appendChild(td);
      });
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    wrapper.appendChild(table);
    container.appendChild(wrapper);

    return { wrapper, table, tbody };
  },

  renderInPreview(previewContent, tableData) {
    previewContent.innerHTML = '';
    const container = document.createElement('div');
    container.className = 'preview-table-container';
    this.render(container, tableData, { maxHeight: 9999 });
    previewContent.appendChild(container);
  },

  sortTable(tbody, colIndex, indicator) {
    const rows = Array.from(tbody.querySelectorAll('tr'));
    const currentDir = indicator.textContent;
    const asc = currentDir !== '↑';

    rows.sort((a, b) => {
      const aVal = a.children[colIndex]?.textContent || '';
      const bVal = b.children[colIndex]?.textContent || '';
      const aNum = parseFloat(aVal);
      const bNum = parseFloat(bVal);
      if (!isNaN(aNum) && !isNaN(bNum)) return asc ? aNum - bNum : bNum - aNum;
      return asc ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);
    });

    rows.forEach(r => tbody.appendChild(r));

    // Update indicators in same row
    const headerRow = tbody.previousElementSibling?.querySelector('tr');
    if (headerRow) {
      headerRow.querySelectorAll('.sort-indicator').forEach(ind => ind.textContent = '↕');
    }
    indicator.textContent = asc ? '↑' : '↓';
  },

  downloadCSV(tableData, filename) {
    const { columns, rows } = tableData;
    const csvRows = [columns.map(c => this.csvEscape(c)).join(',')];
    rows.forEach(row => csvRows.push(row.map(c => this.csvEscape(String(c))).join(',')));
    const blob = new Blob(['\uFEFF' + csvRows.join('\n')], { type: 'text/csv;charset=utf-8;' });
    this.downloadBlob(blob, filename || 'data.csv');
  },

  downloadXLSX(tableData, filename) {
    if (typeof XLSX === 'undefined') { this.downloadCSV(tableData, filename?.replace('.xlsx', '.csv')); return; }
    const ws = XLSX.utils.aoa_to_sheet([tableData.columns, ...tableData.rows]);
    const wb = XLSX.utils.book_new();
    XLSX.utils.book_append_sheet(wb, ws, 'Sheet1');
    XLSX.writeFile(wb, filename || 'data.xlsx');
  },

  downloadBlob(blob, filename) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = filename;
    document.body.appendChild(a); a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  },

  csvEscape(str) {
    if (str.includes(',') || str.includes('"') || str.includes('\n')) return `"${str.replace(/"/g, '""')}"`;
    return str;
  },
};