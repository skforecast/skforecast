document.addEventListener('DOMContentLoaded', function() {
    const versionSwitchLink = document.getElementById('version-switch-link');
    const currentPath = window.location.pathname;
    
    if (versionSwitchLink && currentPath) {
        // Extract path segments
        const pathParts = currentPath.split('/').filter(Boolean);
        
        // Remove version segment (assuming it's the first part)
        if (pathParts.length > 0 && (pathParts[0].startsWith('v') || pathParts[0] === 'dev' || pathParts[0] === 'latest')) {
            pathParts.shift();
        }
        
        // If we have remaining path segments, try to preserve them
        if (pathParts.length > 0) {
            const pagePathWithoutVersion = pathParts.join('/');
            let stableBaseUrl = versionSwitchLink.href.replace(/\/$/, '');
            versionSwitchLink.href = `${stableBaseUrl}/${pagePathWithoutVersion}/`;
        }
    }
});
