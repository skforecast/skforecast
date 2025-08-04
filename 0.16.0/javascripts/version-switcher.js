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

        // SVG warning icon (centrado verticalmente y mayor tamaño)
        const warningSvg = `
            <svg style="vertical-align:middle; margin-right:1em; flex-shrink:0;" width="32" height="32" viewBox="0 0 24 24" fill="none">
                <circle cx="12" cy="12" r="12" fill="#fff"/>
                <path d="M12 7v6M12 17h.01" stroke="#F79939" stroke-width="2.2" stroke-linecap="round"/>
            </svg>
        `;

        // Banner layout: icono y texto alineados
        banner.innerHTML = `
            <div style="display: flex; align-items: center; gap: 1em;">
                ${warningSvg}
                <span style="font-size: 1.25em; font-weight: 600; line-height:1.3;">
                    ${message}
                </span>
            </div>
        `;

        banner.setAttribute('role', 'alert');
        banner.style.position = 'fixed';
        banner.style.top = '32px';
        banner.style.left = '50%';
        banner.style.transform = 'translateX(-50%)';
        banner.style.background = "#f79939";
        banner.style.color = "#001633";
        banner.style.padding = '1.35em 2.2em';
        banner.style.borderRadius = '14px';
        banner.style.boxShadow = '0 2px 18px rgba(0,0,0,0.16)';
        banner.style.zIndex = 10000;
        banner.style.fontWeight = 'bold';
        banner.style.fontFamily = 'inherit';
        banner.style.textAlign = 'center';
        banner.style.maxWidth = "95vw";
        banner.style.opacity = 0;
        banner.style.transition = 'opacity 0.4s';
        banner.style.fontSize = "1.3em";

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
                  4000 // banner duration
                );
                setTimeout(() => window.location.href = "/latest/", 4200);  // Redirect after banner + 200ms
            }
        }).catch(() => {
            window.location.href = "/latest/";
        });
    };
});
