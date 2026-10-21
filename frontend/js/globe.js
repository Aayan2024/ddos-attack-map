import { CONFIG } from './config.js';

const scriptPromises = new Map();

function loadScript(url) {
    if (scriptPromises.has(url)) {
        return scriptPromises.get(url);
    }

    const promise = new Promise((resolve, reject) => {
        const script = document.createElement('script');

        script.src = url;

        script.onload = () => resolve();

        script.onerror = () => reject(
            new Error(`Could not load graphics dependency: ${url}`)
        );

        document.head.appendChild(script);
    });

    scriptPromises.set(url, promise);
    return promise;
}

async function loadGraphicsDependencies() {
    if (!window.THREE) {
        await loadScript(CONFIG.THREE_URL);
    }

    if (!window.THREE?.OrbitControls) {
        await loadScript(CONFIG.ORBIT_CONTROLS_URL);
    }

    if (!window.THREE?.OrbitControls) {
        throw new Error('Three.js or OrbitControls is unavailable');
    }

    return window.THREE;
}

export function hasValidLocation(data) {
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

function colorFor(data) {
    if (data.kind === 'sample') return 0x57bdff;

    const score = data.abuseConfidence;

    if (!Number.isFinite(score) || score < 0 || score > 100) {
        return 0xa7b2c8;
    }

    if (score >= 95) return 0xff4564;
    if (score >= 90) return 0xff8651;
    return 0xf5c65d;
}

export async function createGlobe({
    canvas,
    onSelect = () => {},
    onStatistics = () => {},
    onFPS = () => {},
    onLog = () => {},
    onFatal = () => {}
}) {
    const THREE = await loadGraphicsDependencies();

    const scene = new THREE.Scene();

    const camera = new THREE.PerspectiveCamera(
        55,
        window.innerWidth / window.innerHeight,
        0.1,
        1000
    );

    camera.position.set(0, 0, 3.1);

    const renderer = new THREE.WebGLRenderer({
        canvas,
        antialias: true,
        alpha: false
    });

    renderer.setPixelRatio(
        Math.min(window.devicePixelRatio || 1, 2)
    );

    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setClearColor(0x080d1c);
    renderer.outputEncoding = THREE.sRGBEncoding;

    const controls = new THREE.OrbitControls(camera, canvas);
    controls.enableDamping = true;
    controls.dampingFactor = 0.06;
    controls.enablePan = false;
    controls.minDistance = 1.4;
    controls.maxDistance = 6;
    controls.autoRotate = false;

    const earthMaterial = new THREE.MeshPhongMaterial({
        color: 0x34719a,
        shininess: 8,
        emissive: 0x07152b,
        emissiveIntensity: 0.2
    });

    const earth = new THREE.Mesh(
        new THREE.SphereGeometry(1, 64, 64),
        earthMaterial
    );

    scene.add(earth);

    let destroyed = false;

    new THREE.TextureLoader().load(
        CONFIG.EARTH_TEXTURE_URL,
        texture => {
            if (destroyed) {
                texture.dispose();
                return;
            }

            texture.encoding = THREE.sRGBEncoding;
            earthMaterial.map = texture;
            earthMaterial.color.set(0xffffff);
            earthMaterial.needsUpdate = true;

            onLog('Earth texture loaded');
        },
        undefined,
        () => {
            if (!destroyed) {
                onLog(
                    'Earth texture unavailable; using a plain globe.',
                    true
                );
            }
        }
    );

    scene.add(new THREE.AmbientLight(0xffffff, 0.72));

    const sunlight = new THREE.DirectionalLight(0xffffff, 0.9);
    sunlight.position.set(4, 3, 5);
    scene.add(sunlight);

    scene.add(new THREE.Mesh(
        new THREE.SphereGeometry(1.002, 24, 12),
        new THREE.MeshBasicMaterial({
            color: 0x7ab8e5,
            wireframe: true,
            transparent: true,
            opacity: 0.08
        })
    ));

    const starPositions = new Float32Array(900 * 3);

    for (let index = 0; index < starPositions.length; index += 3) {
        const theta = Math.random() * Math.PI * 2;
        const z = Math.random() * 2 - 1;
        const radius = 25 + Math.random() * 35;
        const radial = Math.sqrt(1 - z * z);

        starPositions[index] = radius * radial * Math.cos(theta);
        starPositions[index + 1] = radius * z;
        starPositions[index + 2] = radius * radial * Math.sin(theta);
    }

    const starGeometry = new THREE.BufferGeometry();

    starGeometry.setAttribute(
        'position',
        new THREE.BufferAttribute(starPositions, 3)
    );

    scene.add(new THREE.Points(
        starGeometry,
        new THREE.PointsMaterial({
            color: 0xc5d8f5,
            size: 0.075,
            transparent: true,
            opacity: 0.8
        })
    ));

    const markers = new Map();
    const selectableMeshes = [];
    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2();

    let pinnedIP = null;
    let hoveredMesh = null;
    let pointerInside = false;
    let pointerDirty = false;
    let pointerStart = null;
    let lastPickTime = 0;
    let animationID = null;
    let frameCount = 0;
    let lastFPSUpdate = performance.now();

    function positionFor(latitude, longitude) {
        const radius = 1.012;
        const phi = (90 - latitude) * Math.PI / 180;
        const theta = (longitude + 180) * Math.PI / 180;

        return new THREE.Vector3(
            -radius * Math.sin(phi) * Math.cos(theta),
            radius * Math.cos(phi),
            radius * Math.sin(phi) * Math.sin(theta)
        );
    }

    function disposeObject(object) {
        object.traverse(child => {
            child.geometry?.dispose();

            if (child.material) {
                const materials = Array.isArray(child.material)
                    ? child.material
                    : [child.material];

                for (const material of materials) {
                    material.map?.dispose();
                    material.dispose();
                }
            }
        });
    }

    function emitStatistics() {
        onStatistics(
            [...markers.values()].map(entry => entry.data)
        );
    }

    function syncSelection() {
        const ip = pinnedIP || hoveredMesh?.userData.ip;
        onSelect(ip ? markers.get(ip)?.data || null : null);
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
        if (!hasValidLocation(data)) return false;

        const previous = markers.get(data.ip);

        if (
            previous?.data.kind === 'indicator' &&
            data.kind === 'sample'
        ) {
            return false;
        }

        removeMarker(data.ip);

        const color = colorFor(data);
        const position = positionFor(data.latitude, data.longitude);
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

    function addRecords(records) {
        let plotted = 0;
        let missing = 0;
        let skipped = 0;

        for (const record of records) {
            if (!hasValidLocation(record)) {
                missing++;
            } else if (upsertMarker(record)) {
                plotted++;
            } else {
                skipped++;
            }
        }

        emitStatistics();
        syncSelection();

        return { plotted, missing, skipped };
    }

    function closeSelection() {
        pinnedIP = null;
        hoveredMesh = null;
        pointerDirty = false;
        canvas.style.cursor = '';
        onSelect(null);
    }

    function clear() {
        for (const ip of [...markers.keys()]) {
            removeMarker(ip);
        }

        closeSelection();
        emitStatistics();
    }

    function pickMarker() {
        if (!pointerInside) return null;

        raycaster.setFromCamera(mouse, camera);

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
        const bounds = canvas.getBoundingClientRect();

        mouse.x =
            ((event.clientX - bounds.left) / bounds.width) * 2 - 1;

        mouse.y =
            -((event.clientY - bounds.top) / bounds.height) * 2 + 1;

        pointerInside = true;
        pointerDirty = true;
    }

    function onPointerLeave() {
        pointerInside = false;
        hoveredMesh = null;
        canvas.style.cursor = '';
        syncSelection();
    }

    function onPointerDown(event) {
        updateMouse(event);
        pointerStart = { x: event.clientX, y: event.clientY };
    }

    function onPointerUp(event) {
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
        syncSelection();
    }

    function onPointerCancel() {
        pointerStart = null;
    }

    function onControlsChange() {
        pointerDirty = true;
    }

    function onResize() {
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();

        renderer.setSize(window.innerWidth, window.innerHeight);
        renderer.setPixelRatio(
            Math.min(window.devicePixelRatio || 1, 2)
        );

        pointerDirty = true;
    }

    function onContextLost(event) {
        event.preventDefault();
        window.cancelAnimationFrame(animationID);

        onFatal(
            'The browser lost its graphics context. ' +
            'Reload the page to restore the globe.'
        );
    }

    canvas.addEventListener('pointermove', updateMouse);
    canvas.addEventListener('pointerleave', onPointerLeave);
    canvas.addEventListener('pointerdown', onPointerDown);
    canvas.addEventListener('pointerup', onPointerUp);
    canvas.addEventListener('pointercancel', onPointerCancel);
    canvas.addEventListener('webglcontextlost', onContextLost);

    controls.addEventListener('change', onControlsChange);
    window.addEventListener('resize', onResize);

    function animate(now) {
        if (destroyed) return;

        animationID = requestAnimationFrame(animate);
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
            canvas.style.cursor = hoveredMesh ? 'pointer' : '';

            syncSelection();

            pointerDirty = false;
            lastPickTime = now;
        }

        renderer.render(scene, camera);
        frameCount++;

        if (now - lastFPSUpdate >= 1000) {
            onFPS(Math.round(
                frameCount * 1000 / (now - lastFPSUpdate)
            ));

            frameCount = 0;
            lastFPSUpdate = now;
        }
    }

    animationID = requestAnimationFrame(animate);
    emitStatistics();

    function destroy() {
        if (destroyed) return;
        destroyed = true;

        window.cancelAnimationFrame(animationID);

        window.removeEventListener('resize', onResize);
        controls.removeEventListener('change', onControlsChange);

        canvas.removeEventListener('pointermove', updateMouse);
        canvas.removeEventListener('pointerleave', onPointerLeave);
        canvas.removeEventListener('pointerdown', onPointerDown);
        canvas.removeEventListener('pointerup', onPointerUp);
        canvas.removeEventListener('pointercancel', onPointerCancel);
        canvas.removeEventListener('webglcontextlost', onContextLost);

        controls.dispose();
        disposeObject(scene);

        markers.clear();
        selectableMeshes.length = 0;
        renderer.dispose();
    }

    return Object.freeze({
        addRecords,
        clear,
        closeSelection,
        destroy
    });
}
