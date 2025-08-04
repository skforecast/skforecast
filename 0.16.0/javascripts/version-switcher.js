(function() {
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

    link.onclick = function(e) {
        e.preventDefault();

        fetch(candidateUrl, { method: 'HEAD' }).then(r => {
            if (r.ok) {
                window.location.href = candidateUrl;
            } else {
                alert("This page does not exist in the latest documentation version. You will be redirected to the homepage.");
                window.location.href = "/latest/";
            }
        }).catch(() => {
            window.location.href = "/latest/";
        });
    };
})();
