document.addEventListener('DOMContentLoaded', function() {
    const link = document.getElementById('version-switch-link');
    if (!link) return;

    const stableVersion = "latest";
    const currentPath = window.location.pathname;
    const pathParts = currentPath.split('/').filter(Boolean);

    let candidateUrl = "/latest/";
    if (pathParts.length > 1 && /^\d+\.\d+\.\d+$/.test(pathParts[0])) {
        pathParts[0] = stableVersion;
        candidateUrl = '/' + pathParts.join('/') + '/';
    }

    link.href = candidateUrl;

    // Banner notification function
    function showBanner(message, duration = 4000) {
        const prev = document.getElementById('skf-version-banner');
        if (prev) prev.remove();

        const banner = document.createElement('div');
        banner.id = 'skf-version-banner';
        // Detect theme (Material)
        const isDark = document.documentElement.getAttribute("data-md-color-scheme") === "slate";
        banner.innerHTML = `<span style="margin-right: 0.7em; font-size:1.3em; vertical-align:middle;">&#9888;&#65039;</span>${message}`;
        banner.setAttribute('role', 'alert');
        banner.style.position = 'fixed';
        banner.style.top = '24px';
        banner.style.left = '50%';
        banner.style.transform = 'translateX(-50%)';
        banner.style.background = "#f79939";
        banner.style.color = isDark ? "#222" : "#001633";
        banner.style.padding = '1em 2em';
        banner.style.borderRadius = '10px';
        banner.style.boxShadow = '0 2px 12px rgba(0,0,0,0.16)';
        banner.style.zIndex = 10000;
        banner.style.fontWeight = 'bold';
        banner.style.fontSize = '1.05em';
        banner.style.fontFamily = 'inherit';
        banner.style.textAlign = 'center';
        banner.style.maxWidth = "95vw";
        banner.style.opacity = 0;
        banner.style.transition = 'opacity 0.4s';
        document.body.appendChild(banner);
        setTimeout(() => banner.style.opacity = 1, 10); // Fade-in
        setTimeout(() => {
            banner.style.opacity = 0;
            setTimeout(() => banner.remove(), 400);
        }, duration);
    }

    link.onclick = function(e) {
        e.preventDefault();
        fetch(candidateUrl, { method: 'HEAD' }).then(r => {
            if (r.ok) {
                window.location.href = candidateUrl;
            } else {
                showBanner(
                  "This page does not exist in the latest documentation version. Redirecting to the home page...", 
                  4000
                );
                setTimeout(() => window.location.href = "/latest/", 4200);
            }
        }).catch(() => {
            window.location.href = "/latest/";
        });
    };
});
