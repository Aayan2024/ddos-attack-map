(() => {
    'use strict';

    // =====================================================
    // Configuration and state
    // =====================================================

    const API_BASE = window.location.origin;
    const AUTO_REFRESH_MS = 300000;

    const markers = new Map();
    const selectableMeshes = [];
    const controllers = new Set();

    const byId = id => document.getElementById(id);

    let scene;
    let camera;
    let renderer;
    let controls;
    let earth;
    let raycaster;

    let dataBusy = false;
    let reportBusy = false;
    let autoRefreshTimer = null;

    let pinnedIP = null;
    let hoveredMesh = null;
    let pointerInsideCanvas = false;
    let pointerDirty = false;
    let pointerStart = null;
    let lastPickTime = 0;

    let frameCount = 0;
    let lastFPSUpdate = performance.now();

    const mouse = { x: 0, y: 0 };

    // =====================================================
    // UI helpers
    // =====================================================

    function debugLog(message, isError = false) {
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

    function setStatus(message, isError = false) {
        const element = byId('operationStatus');
        element.textContent = message;
        element.classList.toggle('error', isError);
    }

    function showFatal(message) {
        byId('fatalError').textContent = message;
        byId('fatalError').hidden = false;
        debugLog(message, true);
    }

    function setDataBusy(value) {
        dataBusy = value;

        document.querySelectorAll('[data-data-action]').forEach(button => {
            button.disabled = value;
        });
    }

    async function runDataAction(label, action) {
        if (dataBusy) return;

        setDataBusy(true);
        setStatus(label);

        try {
            await action();
        } catch (error) {
            setStatus(error.message, true);
            debugLog(error.message, true);
        } finally {
            setDataBusy(false);
        }
    }

    function setHistoryStats(statistics = {}) {
        byId('historyCount').textContent =
            statistics.observation_count ?? '—';

        byId('historyUnique').textContent =
            statistics.unique_ips ?? '—';
    }

    // =====================================================
    // HTTP requests
    // =====================================================

    async function request(
        path,
        { responseType = 'json', timeoutMs = 120000 } = {}
    ) {
        const controller = new AbortController();
        controllers.add(controller);

        const timeout = window.setTimeout(
            () => controller.abort(),
            timeoutMs
        );

        try {
            const response = await fetch(`${API_BASE}${path}`, {
                method: 'GET',
                signal: controller.signal,
                cache: 'no-store'
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
                    // Keep the HTTP status when no JSON error is available.
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
                throw new Error(data.error || 'Request was unsuccessful');
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
            window.clearTimeout(timeout);
            controllers.delete(controller);
        }
    }

    // =====================================================
    // Globe initialization
    // =====================================================

    function initializeGlobe() {
        if (!window.THREE || !window.THREE.OrbitControls) {
            throw new Error(
                'Three.js or OrbitControls failed to load. ' +
                'Check your internet connection and CDN access.'
            );
        }

        scene = new THREE.Scene();

        camera = new THREE.PerspectiveCamera(
            55,
            window.innerWidth / window.innerHeight,
            0.1,
            1000
        );

        camera.position.set(0, 0, 3.1);

        renderer = new THREE.WebGLRenderer({
            canvas: byId('globeCanvas'),
            antialias: true,
            alpha: false
        });

        renderer.setPixelRatio(
            Math.min(window.devicePixelRatio || 1, 2)
        );

        renderer.setSize(window.innerWidth, window.innerHeight);
        renderer.setClearColor(0x080d1c);
        renderer.outputEncoding = THREE.sRGBEncoding;

        controls = new THREE.OrbitControls(
            camera,
            renderer.domElement
        );

        controls.enableDamping = true;
        controls.dampingFactor = 0.06;
        controls.enablePan = false;
        controls.minDistance = 1.4;
        controls.maxDistance = 6;
        controls.autoRotate = false;

        raycaster = new THREE.Raycaster();

        const material = new THREE.MeshPhongMaterial({
            color: 0x34719a,
            shininess: 8,
            emissive: 0x07152b,
            emissiveIntensity: 0.2
        });

        earth = new THREE.Mesh(
            new THREE.SphereGeometry(1, 64, 64),
            material
        );

        scene.add(earth);

        new THREE.TextureLoader().load(
            'https://cdn.jsdelivr.net/npm/three@0.128.0/' +
            'examples/textures/planets/earth_atmos_2048.jpg',
            texture => {
                texture.encoding = THREE.sRGBEncoding;
                material.map = texture;
                material.color.set(0xffffff);
                material.needsUpdate = true;
                debugLog('Earth texture loaded');
            },
            undefined,
            () => {
                debugLog(
                    'Earth texture unavailable; using a plain globe.',
                    true
                );
            }
        );

        scene.add(new THREE.AmbientLight(0xffffff, 0.72));

        const sunlight = new THREE.DirectionalLight(0xffffff, 0.9);
        sunlight.position.set(4, 3, 5);
        scene.add(sunlight);

        const grid = new THREE.Mesh(
            new THREE.SphereGeometry(1.002, 24, 12),
            new THREE.MeshBasicMaterial({
                color: 0x7ab8e5,
                wireframe: true,
                transparent: true,
                opacity: 0.08
            })
        );

        scene.add(grid);
        createStars();
        installPointerHandlers();

        controls.addEventListener('change', () => {
            pointerDirty = true;
        });

        window.addEventListener('resize', () => {
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();

            renderer.setSize(window.innerWidth, window.innerHeight);
            renderer.setPixelRatio(
                Math.min(window.devicePixelRatio || 1, 2)
            );

            pointerDirty = true;
        });

        renderer.domElement.addEventListener(
            'webglcontextlost',
            event => {
                event.preventDefault();
                showFatal(
                    'The browser lost its graphics context. ' +
                    'Reload the page to restore the globe.'
                );
            }
        );

        requestAnimationFrame(animate);
    }

    function createStars() {
        const positions = new Float32Array(900 * 3);

        for (let index = 0; index < positions.length; index += 3) {
            const theta = Math.random() * Math.PI * 2;
            const z = Math.random() * 2 - 1;
            const radius = 25 + Math.random() * 35;
            const radial = Math.sqrt(1 - z * z);

            positions[index] = radius * radial * Math.cos(theta);
            positions[index + 1] = radius * z;
            positions[index + 2] = radius * radial * Math.sin(theta);
        }

        const geometry = new THREE.BufferGeometry();

        geometry.setAttribute(
            'position',
            new THREE.BufferAttribute(positions, 3)
        );

        const material = new THREE.PointsMaterial({
            color: 0xc5d8f5,
            size: 0.075,
            transparent: true,
            opacity: 0.8
        });

        scene.add(new THREE.Points(geometry, material));
    }

    function latLngToXYZ(latitude, longitude, radius = 1.012) {
        const phi = (90 - latitude) * Math.PI / 180;
        const theta = (longitude + 180) * Math.PI / 180;

        return new THREE.Vector3(
            -radius * Math.sin(phi) * Math.cos(theta),
            radius * Math.cos(phi),
            radius * Math.sin(phi) * Math.sin(theta)
        );
    }

    // =====================================================
    // Marker lifecycle
    // =====================================================

    function validLocation(data) {
        return (
            typeof data.ip === 'string' &&
            Number.isFinite(data.latitude) &&
            Number.isFinite(data.longitude) &&
            data.latitude >= -90 &&
            data.latitude <= 90 &&
            data.longitude >= -180 &&
            data.longitude <= 180
        );
    }

    function reputationScore(data) {
        const score = data.abuseConfidence;

        return (
            Number.isFinite(score) &&
            score >= 0 &&
            score <= 100
        ) ? score : null;
    }

    function markerColor(data) {
        if (data.kind === 'sample') return 0x57bdff;

        const score = reputationScore(data);

        if (score === null) return 0xa7b2c8;
        if (score >= 95) return 0xff4564;
        if (score >= 90) return 0xff8651;
        return 0xf5c65d;
    }

    function disposeObject(object) {
        object.traverse(child => {
            child.geometry?.dispose();

            if (child.material) {
                const materials = Array.isArray(child.material)
                    ? child.material
                    : [child.material];

                materials.forEach(material => material.dispose());
            }
        });
    }

    function removeMarker(ip) {
        const entry = markers.get(ip);
        if (!entry) return;

        if (hoveredMesh === entry.mesh) {
            hoveredMesh = null;
        }

        const index = selectableMeshes.indexOf(entry.mesh);

        if (index !== -1) {
            selectableMeshes.splice(index, 1);
        }

        scene.remove(entry.group);
        disposeObject(entry.group);
        markers.delete(ip);
    }

    function upsertMarker(data) {
        if (!validLocation(data)) return false;

        const previous = markers.get(data.ip);

        if (
            previous?.data.kind === 'indicator' &&
            data.kind === 'sample'
        ) {
            return false;
        }

        removeMarker(data.ip);

        const color = markerColor(data);
        const position = latLngToXYZ(data.latitude, data.longitude);
        const outward = position.clone().normalize();
        const group = new THREE.Group();

        const mesh = new THREE.Mesh(
            new THREE.SphereGeometry(0.014, 12, 12),
            new THREE.MeshBasicMaterial({ color })
        );

        mesh.position.copy(position);
        mesh.userData = { ip: data.ip };
        group.add(mesh);

        const glow = new THREE.Mesh(
            new THREE.SphereGeometry(0.027, 10, 10),
            new THREE.MeshBasicMaterial({
                color,
                transparent: true,
                opacity: 0.25,
                depthWrite: false
            })
        );

        glow.position.copy(position);
        group.add(glow);

        const ring = new THREE.Mesh(
            new THREE.RingGeometry(0.022, 0.028, 24),
            new THREE.MeshBasicMaterial({
                color,
                transparent: true,
                opacity: 0.45,
                side: THREE.DoubleSide,
                depthWrite: false
            })
        );

        ring.position.copy(position);

        ring.quaternion.setFromUnitVectors(
            new THREE.Vector3(0, 0, 1),
            outward
        );

        group.add(ring);

        const height = data.kind === 'sample' ? 0.055 : 0.13;

        const beam = new THREE.Mesh(
            new THREE.CylinderGeometry(0.003, 0.006, height, 8),
            new THREE.MeshBasicMaterial({
                color,
                transparent: true,
                opacity: 0.55,
                depthWrite: false
            })
        );

        beam.position.copy(position).addScaledVector(
            outward,
            height / 2
        );

        beam.quaternion.setFromUnitVectors(
            new THREE.Vector3(0, 1, 0),
            outward
        );

        group.add(beam);
        scene.add(group);

        markers.set(data.ip, {
            group,
            mesh,
            glow,
            ring,
            beam,
            data,
            spawnTime: performance.now()
        });

        selectableMeshes.push(mesh);
        pointerDirty = true;

        return true;
    }

    function clearDisplay() {
        for (const ip of [...markers.keys()]) {
            removeMarker(ip);
        }

        pinnedIP = null;
        hoveredMesh = null;

        byId('detailsPanel').hidden = true;
        renderer.domElement.style.cursor = '';

        updateStats();
    }

    function updateStats() {
        const entries = [...markers.values()];

        const indicators = entries.filter(
            entry => entry.data.kind === 'indicator'
        );

        const samples = entries.filter(
            entry => entry.data.kind === 'sample'
        );

        const countries = new Set(
            entries
                .map(entry => entry.data.country)
                .filter(country => country && country !== 'Unknown')
        );

        byId('markerCount').textContent = entries.length;
        byId('indicatorCount').textContent = indicators.length;
        byId('sampleCount').textContent = samples.length;
        byId('countryCount').textContent = countries.size;

        byId('mlHighRisk').textContent = indicators.filter(
            entry => entry.data.ml_prediction?.risk_level === 'High'
        ).length;
    }

    // =====================================================
    // Selection and details
    // =====================================================

    function renderDetails(data) {
        const sample = data.kind === 'sample';
        const score = reputationScore(data);
        const prediction = data.ml_prediction;

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
                : score === null ? 'Not available' : `${score}/100`,

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

    function syncDetails() {
        const ip = pinnedIP || hoveredMesh?.userData.ip;
        const entry = ip ? markers.get(ip) : null;

        if (entry) {
            renderDetails(entry.data);
        } else {
            byId('detailsPanel').hidden = true;
        }
    }

    function pickMarker() {
        if (!pointerInsideCanvas) return null;

        raycaster.setFromCamera(mouse, camera);

        // Earth is included to block picking markers on its far side.
        const hits = raycaster.intersectObjects(
            [earth, ...selectableMeshes],
            false
        );

        if (!hits.length || hits[0].object === earth) {
            return null;
        }

        return hits[0].object;
    }

    function updateMouse(event) {
        const bounds = renderer.domElement.getBoundingClientRect();

        mouse.x =
            ((event.clientX - bounds.left) / bounds.width) * 2 - 1;

        mouse.y =
            -((event.clientY - bounds.top) / bounds.height) * 2 + 1;

        pointerInsideCanvas = true;
        pointerDirty = true;
    }

    function installPointerHandlers() {
        const canvas = renderer.domElement;

        canvas.addEventListener('pointermove', updateMouse);

        canvas.addEventListener('pointerleave', () => {
            pointerInsideCanvas = false;
            hoveredMesh = null;
            canvas.style.cursor = '';
            syncDetails();
        });

        canvas.addEventListener('pointerdown', event => {
            updateMouse(event);

            pointerStart = {
                x: event.clientX,
                y: event.clientY
            };
        });

        canvas.addEventListener('pointerup', event => {
            if (!pointerStart) return;

            const moved = Math.hypot(
                event.clientX - pointerStart.x,
                event.clientY - pointerStart.y
            );

            pointerStart = null;

            if (moved > 6) return;

            updateMouse(event);
            scene.updateMatrixWorld(true);
            camera.updateMatrixWorld(true);

            hoveredMesh = pickMarker();
            pinnedIP = hoveredMesh?.userData.ip || null;
            syncDetails();
        });

        canvas.addEventListener('pointercancel', () => {
            pointerStart = null;
        });
    }

    // =====================================================
    // Backend health and statistics
    // =====================================================

    async function checkBackend() {
        const dot = byId('connectionDot');

        try {
            await request('/', { timeoutMs: 10000 });

            dot.classList.add('connected');
            dot.classList.remove('disconnected');

            byId('connectionText').textContent = 'Backend connected';
            debugLog('Backend connected');

        } catch (error) {
            dot.classList.add('disconnected');
            dot.classList.remove('connected');

            byId('connectionText').textContent =
                'Backend unavailable or misconfigured';

            debugLog(error.message, true);
        }
    }

    async function refreshHistoryStats() {
        try {
            const result = await request(
                '/api/observations?days=7&limit=1',
                { timeoutMs: 15000 }
            );

            setHistoryStats(result.statistics);

        } catch (error) {
            setHistoryStats();
            debugLog(`History statistics: ${error.message}`, true);
        }
    }

    // =====================================================
    // Collection, samples, and history
    // =====================================================

    async function addLocationSamples() {
        await runDataAction('Looking up location samples…', async () => {
            const ips = [
                '8.8.8.8',
                '1.1.1.1',
                '77.88.8.8',
                '180.76.76.76',
                '208.67.222.222'
            ];

            const result = await request(
                '/batch-geolocate?ips=' +
                encodeURIComponent(ips.join(','))
            );

            let added = 0;
            let skipped = 0;

            for (const item of result.results || []) {
                if (
                    !item.error &&
                    upsertMarker({
                        ...item,
                        kind: 'sample',
                        abuseConfidence: null,
                        ml_prediction: null,
                        observed_at: null
                    })
                ) {
                    added++;
                } else {
                    skipped++;
                }
            }

            updateStats();
            syncDetails();

            setStatus(
                `Loaded ${added} location samples; ${skipped} skipped. ` +
                'Samples are not threat findings or stored observations.'
            );

            debugLog(`Location samples loaded: ${added}`);
        });
    }

    async function collectIndicators(withML = false) {
        await runDataAction(
            withML
                ? 'Collecting indicators with experimental ML…'
                : 'Collecting reported indicators…',
            async () => {
                const endpoint = withML
                    ? '/fetch-attacks-ml?limit=20'
                    : '/fetch-attacks?limit=20';

                const result = await request(endpoint);

                if (
                    result.success !== true ||
                    !Array.isArray(result.attacks)
                ) {
                    throw new Error('Unexpected collection response');
                }

                let plotted = 0;
                let missing = 0;

                for (const item of result.attacks) {
                    const record = {
                        ...item,
                        kind: 'indicator',
                        observed_at:
                            result.collection?.observed_at || null
                    };

                    if (!validLocation(record)) {
                        missing++;
                        debugLog(
                            `Stored ${record.ip}; no usable map location`
                        );
                        continue;
                    }

                    if (upsertMarker(record)) plotted++;
                }

                updateStats();
                syncDetails();

                if (result.collection?.observed_at) {
                    byId('lastCollection').textContent =
                        new Date(
                            result.collection.observed_at
                        ).toLocaleTimeString();
                }

                await refreshHistoryStats();

                const message =
                    `Collected ${result.count} reputation indicators. ` +
                    `Updated ${plotted} markers; ` +
                    `${missing} without usable coordinates.`;

                setStatus(message);
                debugLog(message);

                if (result.ml_warning) {
                    debugLog(result.ml_warning);
                }
            }
        );
    }

    async function loadStoredObservations() {
        await runDataAction(
            'Loading stored observations from the last seven days…',
            async () => {
                const result = await request(
                    '/api/observations?days=7&limit=5000'
                );

                if (!Array.isArray(result.items)) {
                    throw new Error('Unexpected history response');
                }

                const latestByIP = new Map();

                // The API returns newest observations first.
                for (const item of result.items) {
                    if (!latestByIP.has(item.ip)) {
                        latestByIP.set(item.ip, item);
                    }
                }

                let plotted = 0;

                for (const item of latestByIP.values()) {
                    if (upsertMarker({
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
                    })) {
                        plotted++;
                    }
                }

                updateStats();
                syncDetails();
                setHistoryStats(result.statistics);

                const capped =
                    (result.statistics?.observation_count || 0) >
                    result.items.length;

                const message =
                    `Loaded ${plotted} markers from stored observations.` +
                    (capped
                        ? ' Based on the newest 5,000 rows.'
                        : '');

                setStatus(message);
                debugLog(message);
            }
        );
    }

    // =====================================================
    // Report download
    // =====================================================

    async function downloadReport() {
        if (reportBusy) return;

        const days = Number(byId('reportDays').value);

        if (!Number.isInteger(days) || days < 1 || days > 90) {
            setStatus('Choose a whole number from 1 to 90 days.', true);
            return;
        }

        reportBusy = true;
        byId('reportBtn').disabled = true;
        byId('reportDays').disabled = true;

        setStatus(`Preparing the ${days}-day workbook…`);

        try {
            const { blob, response } = await request(
                `/api/reports/weekly.xlsx?days=${days}`,
                { responseType: 'blob' }
            );

            const disposition =
                response.headers.get('Content-Disposition') || '';

            const match = disposition.match(/filename="([^"]+)"/i);

            const filename = match
                ? match[1]
                : `intelligence-report-${days}-days.xlsx`;

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

            setStatus(
                'Workbook download started. It contains stored ' +
                'observations, not reconstructed historical attacks.'
            );

            debugLog(`Workbook downloaded: ${filename}`);

        } catch (error) {
            setStatus(error.message, true);
            debugLog(`Report: ${error.message}`, true);

        } finally {
            reportBusy = false;
            byId('reportBtn').disabled = false;
            byId('reportDays').disabled = false;
        }
    }

    // =====================================================
    // Browser-driven refresh
    // =====================================================

    function toggleAutoRefresh() {
        const button = byId('autoRefreshBtn');

        if (autoRefreshTimer !== null) {
            window.clearInterval(autoRefreshTimer);
            autoRefreshTimer = null;

            button.textContent = 'Start Auto-Refresh · 5 min';
            button.classList.remove('active');

            setStatus('Auto-refresh stopped.');
            debugLog('Auto-refresh stopped');
            return;
        }

        autoRefreshTimer = window.setInterval(() => {
            if (!dataBusy) {
                collectIndicators(false);
            } else {
                debugLog(
                    'Auto-refresh skipped: another data operation is running'
                );
            }
        }, AUTO_REFRESH_MS);

        button.textContent = 'Stop Auto-Refresh · 5 min';
        button.classList.add('active');

        setStatus(
            'Auto-refresh enabled. Next collection in five minutes. ' +
            'Closing this page stops browser-driven collection.'
        );

        debugLog('Auto-refresh enabled: five-minute interval');
    }

    // =====================================================
    // Rendering loop
    // =====================================================

    function animate(now) {
        requestAnimationFrame(animate);
        controls.update();

        for (const entry of markers.values()) {
            const seconds = (now - entry.spawnTime) / 1000;
            const phase = (seconds % 2) / 2;

            entry.glow.scale.setScalar(
                1 + Math.sin(seconds * 3) * 0.15
            );

            entry.ring.scale.setScalar(1 + phase * 2);
            entry.ring.material.opacity = 0.45 * (1 - phase);

            entry.beam.material.opacity =
                0.45 + Math.sin(seconds * 3) * 0.12;
        }

        scene.updateMatrixWorld(true);
        camera.updateMatrixWorld(true);

        if (pointerDirty && now - lastPickTime > 60) {
            hoveredMesh = pickMarker();

            renderer.domElement.style.cursor =
                hoveredMesh ? 'pointer' : '';

            syncDetails();

            pointerDirty = false;
            lastPickTime = now;
        }

        renderer.render(scene, camera);
        frameCount++;

        if (now - lastFPSUpdate >= 1000) {
            byId('fps').textContent = Math.round(
                frameCount * 1000 / (now - lastFPSUpdate)
            );

            frameCount = 0;
            lastFPSUpdate = now;
        }
    }

    // =====================================================
    // Event wiring and startup
    // =====================================================

    function installButtons() {
        byId('samplesBtn').addEventListener(
            'click', addLocationSamples
        );

        byId('fetchBtn').addEventListener(
            'click', () => collectIndicators(false)
        );

        byId('mlBtn').addEventListener(
            'click', () => collectIndicators(true)
        );

        byId('historyBtn').addEventListener(
            'click', loadStoredObservations
        );

        byId('autoRefreshBtn').addEventListener(
            'click', toggleAutoRefresh
        );

        byId('reportBtn').addEventListener(
            'click', downloadReport
        );

        byId('clearBtn').addEventListener('click', () => {
            clearDisplay();

            setStatus(
                'Display cleared. Stored history has not been deleted.'
            );

            debugLog('Display cleared; database unchanged');
        });

        byId('closeDetailsBtn').addEventListener('click', () => {
            pinnedIP = null;
            hoveredMesh = null;
            pointerDirty = false;
            byId('detailsPanel').hidden = true;
        });
    }

    async function start() {
        if (!['http:', 'https:'].includes(location.protocol)) {
            showFatal(
                'Open the website through FastAPI at ' +
                'http://127.0.0.1:8000/globe, not directly from a file.'
            );
            return;
        }

        try {
            initializeGlobe();
            installButtons();

            debugLog('Globe initialized; no indicators auto-loaded');

            await checkBackend();
            await refreshHistoryStats();

        } catch (error) {
            showFatal(`Initialization failed: ${error.message}`);
        }
    }

    window.addEventListener('pagehide', () => {
        if (autoRefreshTimer !== null) {
            window.clearInterval(autoRefreshTimer);
            autoRefreshTimer = null;

            byId('autoRefreshBtn').textContent =
                'Start Auto-Refresh · 5 min';

            byId('autoRefreshBtn').classList.remove('active');
        }

        for (const controller of controllers) {
            controller.abort();
        }
    });

    start();
})();
