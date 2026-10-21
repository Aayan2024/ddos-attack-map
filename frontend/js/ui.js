const byId = id => document.getElementById(id);

export function log(message, isError = false) {
    const line = document.createElement('div');

    line.textContent =
        `[${new Date().toLocaleTimeString()}] ${message}`;

    line.classList.toggle('log-error', isError);

    const panel = byId('debugPanel');
    panel.appendChild(line);

    while (panel.children.length > 100) {
        panel.firstElementChild.remove();
    }

    panel.scrollTop = panel.scrollHeight;

    if (isError) {
        console.error(message);
    } else {
        console.log(message);
    }
}

export function setStatus(message, isError = false) {
    const element = byId('operationStatus');

    element.textContent = message;
    element.classList.toggle('error', isError);
}

export function showFatal(message) {
    byId('fatalError').textContent = message;
    byId('fatalError').hidden = false;
    log(message, true);
}

export function setConnection(connected) {
    const dot = byId('connectionDot');

    dot.classList.toggle('connected', connected);
    dot.classList.toggle('disconnected', !connected);

    byId('connectionText').textContent = connected
        ? 'Backend connected'
        : 'Backend unavailable or misconfigured';
}

export function setDataBusy(busy) {
    document.querySelectorAll('[data-data-action]').forEach(button => {
        button.disabled = busy;
    });
}

export function setReportBusy(busy) {
    byId('reportBtn').disabled = busy;
    byId('reportDays').disabled = busy;
}

export function getReportDays() {
    const days = Number(byId('reportDays').value);

    if (!Number.isInteger(days) || days < 1 || days > 90) {
        throw new Error(
            'Choose a whole number from 1 to 90 days.'
        );
    }

    return days;
}

export function setAutoRefresh(enabled) {
    const button = byId('autoRefreshBtn');

    button.textContent = enabled
        ? 'Stop Auto-Refresh · 5 min'
        : 'Start Auto-Refresh · 5 min';

    button.classList.toggle('active', enabled);
}

export function setHistoryStatistics(statistics = {}) {
    byId('historyCount').textContent =
        statistics.observation_count ?? '—';

    byId('historyUnique').textContent =
        statistics.unique_ips ?? '—';
}

export function setLastCollection(timestamp) {
    if (!timestamp) return;

    const date = new Date(timestamp);

    byId('lastCollection').textContent =
        Number.isNaN(date.getTime())
            ? '—'
            : date.toLocaleTimeString();
}

export function setFPS(value) {
    byId('fps').textContent = value;
}

export function setMarkerStatistics(records) {
    const indicators = records.filter(
        item => item.kind === 'indicator'
    );

    const samples = records.filter(
        item => item.kind === 'sample'
    );

    const countries = new Set(
        records
            .map(item => item.country)
            .filter(country => country && country !== 'Unknown')
    );

    byId('markerCount').textContent = records.length;
    byId('indicatorCount').textContent = indicators.length;
    byId('sampleCount').textContent = samples.length;
    byId('countryCount').textContent = countries.size;

    byId('mlHighRisk').textContent = indicators.filter(
        item => item.ml_prediction?.risk_level === 'High'
    ).length;
}

export function renderDetails(data) {
    if (!data) {
        byId('detailsPanel').hidden = true;
        return;
    }

    const sample = data.kind === 'sample';
    const prediction = data.ml_prediction;

    const score = (
        Number.isFinite(data.abuseConfidence) &&
        data.abuseConfidence >= 0 &&
        data.abuseConfidence <= 100
    ) ? data.abuseConfidence : null;

    const values = {
        infoKind: sample
            ? 'Location sample — not a threat finding'
            : 'Reported reputation indicator',

        infoIP: data.ip,

        infoLocation:
            `${data.city || 'Unknown'}, ${data.region || 'Unknown'}`,

        infoCountry: data.country || 'Unknown',
        infoOrg: data.org || 'Unknown',

        infoCoordinates:
            `${data.latitude.toFixed(4)}, ` +
            `${data.longitude.toFixed(4)} · approximate`,

        infoTimezone: data.timezone || 'Not available',

        infoAbuse: sample
            ? 'Not assessed'
            : score === null
                ? 'Not available'
                : `${score}/100`,

        infoObserved:
            data.observed_at || 'Not recorded for this sample',

        infoMLRisk: prediction
            ? prediction.risk_level || 'Unknown'
            : 'Not evaluated',

        infoMLConf: Number.isFinite(prediction?.confidence)
            ? `${prediction.confidence.toFixed(1)}% · experimental`
            : 'Not available'
    };

    for (const [id, value] of Object.entries(values)) {
        byId(id).textContent = value;
    }

    byId('mlWarning').hidden = !prediction;
    byId('detailsPanel').hidden = false;
}

export function downloadBlob(blob, filename) {
    const objectURL = URL.createObjectURL(blob);
    const link = document.createElement('a');

    link.href = objectURL;
    link.download = filename;

    document.body.appendChild(link);
    link.click();
    link.remove();

    window.setTimeout(
        () => URL.revokeObjectURL(objectURL),
        60000
    );
}

export function bindActions(actions) {
    const bindings = {
        samplesBtn: actions.samples,
        fetchBtn: actions.collect,
        mlBtn: actions.collectML,
        historyBtn: actions.history,
        autoRefreshBtn: actions.autoRefresh,
        clearBtn: actions.clear,
        reportBtn: actions.report,
        closeDetailsBtn: actions.closeDetails
    };

    for (const [id, handler] of Object.entries(bindings)) {
        byId(id).addEventListener('click', handler);
    }
}
