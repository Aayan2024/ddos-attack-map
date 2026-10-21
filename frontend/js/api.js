import { CONFIG } from './config.js';

const activeControllers = new Set();

export function cancelRequests() {
    for (const controller of activeControllers) {
        controller.abort();
    }
}

export async function request(
    path,
    {
        responseType = 'json',
        timeoutMs = CONFIG.REQUEST_TIMEOUT_MS
    } = {}
) {
    const controller = new AbortController();
    activeControllers.add(controller);

    const timer = window.setTimeout(
        () => controller.abort(),
        timeoutMs
    );

    try {
        const response = await fetch(`${CONFIG.API_BASE}${path}`, {
            method: 'GET',
            cache: 'no-store',
            signal: controller.signal
        });

        if (!response.ok) {
            let message = `Request failed: HTTP ${response.status}`;

            try {
                const body = await response.json();

                if (typeof body.detail === 'string') {
                    message = body.detail;
                } else if (Array.isArray(body.detail)) {
                    message = body.detail
                        .map(item => item.msg || 'Invalid request')
                        .join('; ');
                } else if (typeof body.error === 'string') {
                    message = body.error;
                }
            } catch {
                // Preserve HTTP status when the response is not JSON.
            }

            throw new Error(message);
        }

        if (responseType === 'blob') {
            return {
                blob: await response.blob(),
                response
            };
        }

        const data = await response.json();

        if (data && (data.success === false || data.error)) {
            throw new Error(
                data.error || 'The operation was unsuccessful'
            );
        }

        return data;

    } catch (error) {
        if (error.name === 'AbortError') {
            throw new Error(
                'Request timed out or was cancelled. Check the backend.'
            );
        }

        throw error;

    } finally {
        window.clearTimeout(timer);
        activeControllers.delete(controller);
    }
}

export function getHealth() {
    return request('/health', { timeoutMs: 10000 });
}

export function getLocationSamples() {
    const ips = encodeURIComponent(CONFIG.SAMPLE_IPS.join(','));

    return request(`/batch-geolocate?ips=${ips}`);
}

export function collectIndicators(withML = false) {
    const endpoint = withML
        ? '/fetch-attacks-ml'
        : '/fetch-attacks';

    return request(
        `${endpoint}?limit=${CONFIG.COLLECTION_LIMIT}`
    );
}

export function getHistory({
    days = CONFIG.HISTORY_DAYS,
    limit = CONFIG.HISTORY_LIMIT
} = {}) {
    const query = new URLSearchParams({
        days: String(days),
        limit: String(limit)
    });

    return request(`/api/observations?${query}`);
}

export function getHistoryStatistics() {
    return request(
        `/api/observations?days=${CONFIG.HISTORY_DAYS}&limit=1`,
        { timeoutMs: 15000 }
    );
}

export async function getWorkbook(days) {
    const { blob, response } = await request(
        `/api/reports/weekly.xlsx?days=${days}`,
        { responseType: 'blob' }
    );

    const disposition =
        response.headers.get('Content-Disposition') || '';

    const match = disposition.match(/filename="([^"]+)"/i);

    return {
        blob,
        filename: match
            ? match[1]
            : `intelligence-report-${days}-days.xlsx`
    };
}
