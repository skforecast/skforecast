document.addEventListener('DOMContentLoaded', function() {
    const link = document.getElementById('version-switch-link');
    if (!link) return;

    const stableUrl = new URL(link.href);
    const stableVersion = stableUrl.pathname.split('/').filter(Boolean)[0] || 'latest';

    const currentPath = window.location.pathname;
    const pathParts = currentPath.split('/').filter(Boolean);

    let targetHref = stableUrl.href;
    if (pathParts.length > 1 && /^\d+\.\d+\.\d+$/.test(pathParts[0])) {
        pathParts[0] = stableVersion;
        targetHref = stableUrl.origin + '/' + pathParts.join('/') + '/';
    }

    link.addEventListener('click', function(e) {
        e.preventDefault(); // Evita la navegación directa
        // Primero, prueba si existe la página en latest
        fetch(targetHref, { method: 'HEAD' })
          .then(response => {
              if (response.ok) {
                  window.location.href = targetHref;
              } else {
                  window.location.href = stableUrl.href; // Home de latest
              }
          })
          .catch(() => {
              window.location.href = stableUrl.href;
          });
    });

    // Opcional: actualiza el href para mostrar la URL real al pasar el ratón
    link.href = targetHref;
});
