document.addEventListener('DOMContentLoaded', function() {
    const link = document.getElementById('version-switch-link');
    if (!link) return;

    // Usa el pathname base de la versión estable como referencia
    const stableUrl = new URL(link.href, window.location.origin);
    const stableVersion = stableUrl.pathname.split('/').filter(Boolean)[0] || 'latest';

    // Obtén el path actual (ej: "/0.16.0/user_guides/algo/")
    const pathParts = window.location.pathname.split('/').filter(Boolean);

    // Si el path tiene un segmento de versión tipo X.Y.Z
    if (pathParts.length > 1 && /^\d+\.\d+\.\d+$/.test(pathParts[0])) {
        pathParts[0] = stableVersion;
        var candidateUrl = '/' + pathParts.join('/') + '/';
    } else {
        var candidateUrl = stableUrl.pathname;
    }

    // Al hacer click, comprobamos si existe el destino
    link.addEventListener('click', function(e) {
        e.preventDefault();
        fetch(candidateUrl, { method: 'HEAD' }).then(r => {
            if (r.ok) {
                window.location.href = candidateUrl;
            } else {
                window.location.href = stableUrl.pathname; // home de latest
            }
        }).catch(() => {
            window.location.href = stableUrl.pathname;
        });
    });

    // Para que al pasar el ratón, el navegador muestre el link real
    link.href = candidateUrl;
});
