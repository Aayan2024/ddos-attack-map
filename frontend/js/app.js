import { CONFIG } from './config.js';
import * as api from './api.js';
import * as ui from './ui.js';
import { createGlobe } from './globe.js';

let globe = null;
let dataBusy = false;
let reportBusy = false;
let autoRefreshTimer = null;

async function runDataAction(label, action) {
    if (dataBusy) return;

    dataBusy = true;
    ui.setDataBusy(true);
    ui.setStatus(label);

    try {
        await action();
    } catch (error) {
        ui.setStatus(error.message, true);
        ui.log(error.message, true);
    } finally {
        dataBusy = false;
        ui.setDataBusy(false);
    }
}

async function checkBackend() {
    try {
        await api.getHealth();
        ui.setConnection(true);
        ui.log('Backend connected');
    } catch (error) {
        ui.setConnection(false);
        ui.log(error.message, true);
    }
}

async function refreshHistoryStatistics() {
    try {
        const result = await api.getHistoryStatistics();
        ui.setHistoryStatistics(result.statistics);
    } catch (error) {
        ui.setHistoryStatistics();
        ui.log(`History statistics: ${error.message}`, true);
    }
}

async function addSamples() {
    await runDataAction(
        'Looking up location samples…',
        async () => {
            const result = await api.getLocationSamples();

            if (!Array.isArray(result.results)) {
                throw new Error('Unexpected location-sample response');
            }

            const successful = result.results.filter(item => !item.error);

            const records = successful.map(item => ({
                ...item,
                kind: 'sample',
                abuseConfidence: null,
                ml_prediction: null,
                observed_at: null
            }));

            const summary = globe.addRecords(records);

            const skipped =
                result.results.length - successful.length +
                summary.missing +
                summary.skipped;

            ui.setStatus(
                `Loaded ${summary.plotted} location samples; ` +
                `${skipped} skipped. Samples are not threat findings.`
            );

            ui.log(`Location samples loaded: ${summary.plotted}`);
        }
    );
}

async function collect(withML = false) {
    await runDataAction(
        withML
            ? 'Collecting indicators with experimental ML…'
            : 'Collecting reported indicators…',
        async () => {
            const result = await api.collectIndicators(withML);

            if (
                result.success !== true ||
                !Array.isArray(result.attacks)
            ) {
                throw new Error('Unexpected collection response');
            }

            const records = result.attacks.map(item => ({
                ...item,
                kind: 'indicator',
                observed_at: result.collection?.observed_at || null
            }));

            const summary = globe.addRecords(records);

            ui.setLastCollection(result.collection?.observed_at);
            await refreshHistoryStatistics();

            const message =
                `Collected ${result.count} reputation indicators. ` +
                `Updated ${summary.plotted} markers; ` +
                `${summary.missing} without usable coordinates.`;

            ui.setStatus(message);
            ui.log(message);

            if (result.ml_warning) {
                ui.log(result.ml_warning);
            }
        }
    );
}

async function loadHistory() {
    await runDataAction(
        'Loading stored observations from the last seven days…',
        async () => {
            const result = await api.getHistory();

            if (!Array.isArray(result.items)) {
                throw new Error('Unexpected history response');
            }

            const latestByIP = new Map();

            // The backend returns newest observations first.
            for (const item of result.items) {
                if (!latestByIP.has(item.ip)) {
                    latestByIP.set(item.ip, item);
                }
            }

            const records = [...latestByIP.values()].map(item => ({
                ip: item.ip,
                latitude: item.latitude,
                longitude: item.longitude,
                country: item.country,
                region: item.region,
                city: item.city,
                org: item.organization,
                abuseConfidence: item.abuse_confidence,
                observed_at: item.observed_at,
                kind: 'indicator',
                ml_prediction: null
            }));

            const summary = globe.addRecords(records);
            ui.setHistoryStatistics(result.statistics);

            const capped =
                (result.statistics?.observation_count || 0) >
                result.items.length;

            const message =
                `Loaded ${summary.plotted} markers from stored observations. ` +
                `${summary.missing} without usable coordinates.` +
                (capped
                    ? ` Based on the newest ${CONFIG.HISTORY_LIMIT} rows.`
                    : '');

            ui.setStatus(message);
            ui.log(message);
        }
    );
}

async function downloadReport() {
    if (reportBusy) return;

    let days;

    try {
        days = ui.getReportDays();
    } catch (error) {
        ui.setStatus(error.message, true);
        return;
    }

    reportBusy = true;
    ui.setReportBusy(true);
    ui.setStatus(`Preparing the ${days}-day workbook…`);

    try {
        const { blob, filename } = await api.getWorkbook(days);
        ui.downloadBlob(blob, filename);

        ui.setStatus(
            'Workbook download started. It contains stored observations, ' +
            'not reconstructed historical attacks.'
        );

        ui.log(`Workbook downloaded: ${filename}`);

    } catch (error) {
        ui.setStatus(error.message, true);
        ui.log(`Report: ${error.message}`, true);

    } finally {
        reportBusy = false;
        ui.setReportBusy(false);
    }
}

function stopAutoRefresh() {
    if (autoRefreshTimer !== null) {
        window.clearInterval(autoRefreshTimer);
        autoRefreshTimer = null;
    }

    ui.setAutoRefresh(false);
}

function toggleAutoRefresh() {
    if (autoRefreshTimer !== null) {
        stopAutoRefresh();
        ui.setStatus('Auto-refresh stopped.');
        ui.log('Auto-refresh stopped');
        return;
    }

    autoRefreshTimer = window.setInterval(() => {
        if (!dataBusy) {
            collect(false);
        } else {
            ui.log(
                'Auto-refresh skipped: another data operation is running'
            );
        }
    }, CONFIG.AUTO_REFRESH_MS);

    ui.setAutoRefresh(true);

    ui.setStatus(
        'Auto-refresh enabled. Next collection in five minutes. ' +
        'Closing this page stops browser-driven collection.'
    );

    ui.log('Auto-refresh enabled: five-minute interval');
}

function clearDisplay() {
    globe.clear();

    ui.setStatus(
        'Display cleared. Stored history has not been deleted.'
    );

    ui.log('Display cleared; database unchanged');
}

async function start() {
    if (!['http:', 'https:'].includes(window.location.protocol)) {
        ui.showFatal(
            'Open the website through FastAPI at ' +
            'http://127.0.0.1:8000/globe, not directly from a file.'
        );
        return;
    }

    // Keep controls disabled until the globe is ready.
    ui.setDataBusy(true);
    ui.setReportBusy(true);
    document.getElementById('autoRefreshBtn').disabled = true;

    try {
        globe = await createGlobe({
            canvas: document.getElementById('globeCanvas'),
            onSelect: ui.renderDetails,
            onStatistics: ui.setMarkerStatistics,
            onFPS: ui.setFPS,
            onLog: ui.log,
            onFatal: ui.showFatal
        });

        ui.bindActions({
            samples: addSamples,
            collect: () => collect(false),
            collectML: () => collect(true),
            history: loadHistory,
            autoRefresh: toggleAutoRefresh,
            clear: clearDisplay,
            report: downloadReport,
            closeDetails: () => globe.closeSelection()
        });

        ui.setDataBusy(false);
        ui.setReportBusy(false);
        document.getElementById('autoRefreshBtn').disabled = false;

        ui.log('Modular frontend initialized; no indicators auto-loaded');

        await Promise.all([
            checkBackend(),
            refreshHistoryStatistics()
        ]);

    } catch (error) {
        ui.showFatal(`Initialization failed: ${error.message}`);
    }
}

window.addEventListener('pagehide', event => {
    stopAutoRefresh();
    api.cancelRequests();

    // Preserve graphics when the browser keeps the page in back/forward cache.
    if (!event.persisted) {
        globe?.destroy();
    }
});

start();
