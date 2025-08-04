/**
 * Version Switcher: para docs con versiones tipo 0.16.0, 1.0.0, etc.
 * Reemplaza el primer segmento del path por la versión del enlace banner (ej. "latest")
 */

document.addEventListener('DOMContentLoaded', function() {
    const link = document.getElementById('version-switch-link');
    if (!link) return;

    // Detecta la versión estable a partir del href del enlace del banner
    const stableUrl = new URL(link.href);
    const stableVersion = stableUrl.pathname.split('/').filter(Boolean)[0] || 'latest';

    // Path actual, ej: "/0.16.0/user_guides/..."
    const currentPath = window.location.pathname;
    const pathParts = currentPath.split('/').filter(Boolean);

    // Si la ruta empieza por algo tipo "0.16.0", "1.0.0", etc.
    if (pathParts.length > 1 && /^\d+\.\d+\.\d+$/.test(pathParts[0])) {
        pathParts[0] = stableVersion;
        const newPath = '/' + pathParts.join('/') + '/';
        link.href = stableUrl.origin + newPath;
    } else {
        // Si no detecta la versión, deja el link tal cual (al home de latest)
        link.href = stableUrl.href;
    }
});
