// Live client-side filter for the Examples card grids.
//
// Progressive enhancement: the filter bar is rendered with the `hidden`
// attribute by tools/docs_hooks/examples_grid.py and only unhidden here, so with
// JavaScript disabled the pages are plain, complete card grids.
//
// The script is a no-op on every page without #ex-filter, and re-initialises on
// instant navigation via document$ (same idiom as version-switcher.js).

(function () {
  function init() {
    const bar = document.querySelector('#ex-filter');
    if (!bar) return;

    const input = bar.querySelector('#ex-q');
    const countEl = bar.querySelector('[data-ex-count]');
    const resetEl = bar.querySelector('[data-ex-reset]');
    const noResEl = document.querySelector('.ex-noresults');
    const chips = Array.from(bar.querySelectorAll('.ex-chip[data-ex-group]'));

    // Localised count templates, emitted by the hook.
    const tplAll = countEl ? countEl.getAttribute('data-ex-count-all') || '' : '';
    const tplSome = countEl ? countEl.getAttribute('data-ex-count-some') || '' : '';

    // Precompute the search haystack once per rendered document.
    const items = Array.from(document.querySelectorAll('li.ex-card')).map(function (li) {
      const grid = li.closest('[data-ex-section]');
      const tags = (li.getAttribute('data-ex-tags') || '').split(/\s+/).filter(Boolean);
      return {
        el: li,
        section: grid ? grid.getAttribute('data-ex-section') : '',
        tags: tags,
        text: (li.textContent + ' ' + tags.join(' ')).toLowerCase()
      };
    });
    if (!items.length) return;
    const total = items.length;

    // Each section key maps to its heading, its intro paragraph and its grid.
    const sections = {};
    items.forEach(function (item) {
      if (item.section && !sections[item.section]) {
        sections[item.section] = Array.from(
          document.querySelectorAll('[data-ex-section="' + item.section + '"]')
        );
      }
    });

    // State is local to this init call, so nothing leaks across navigations.
    const active = { level: new Set(), topic: new Set() };

    function matches(item, query) {
      for (const group in active) {
        const selected = active[group];
        if (!selected.size) continue; // an untouched group adds no constraint
        // OR within a group, AND across groups.
        if (!item.tags.some(function (tag) { return selected.has(tag); })) return false;
      }
      return !query || item.text.indexOf(query) !== -1;
    }

    let countTimer = null;

    function apply() {
      const query = (input.value || '').trim().toLowerCase();
      const populated = Object.create(null);
      let shown = 0;

      items.forEach(function (item) {
        const visible = matches(item, query);
        item.el.hidden = !visible;
        if (visible) {
          shown += 1;
          populated[item.section] = true;
        }
      });

      Object.keys(sections).forEach(function (key) {
        const visible = !!populated[key];
        sections[key].forEach(function (el) { el.hidden = !visible; });
      });

      const filtered = !!query || active.level.size > 0 || active.topic.size > 0;
      if (resetEl) resetEl.hidden = !filtered;
      if (noResEl) noResEl.hidden = shown !== 0;

      // Debounced so that a screen reader announces the settled count once
      // rather than once per keystroke.
      if (countEl) {
        clearTimeout(countTimer);
        countTimer = setTimeout(function () {
          const tpl = shown === total ? tplAll : tplSome;
          countEl.textContent = tpl
            .replace('{shown}', shown)
            .replace('{total}', total);
        }, 250);
      }
    }

    function reset() {
      input.value = '';
      active.level.clear();
      active.topic.clear();
      chips.forEach(function (chip) { chip.setAttribute('aria-pressed', 'false'); });
      apply();
    }

    bar.addEventListener('input', function (event) {
      if (event.target === input) apply();
    });

    bar.addEventListener('click', function (event) {
      const chip = event.target.closest('.ex-chip[data-ex-group]');
      if (chip) {
        const selected = active[chip.dataset.exGroup];
        if (!selected) return;
        const pressed = chip.getAttribute('aria-pressed') !== 'true';
        chip.setAttribute('aria-pressed', pressed ? 'true' : 'false');
        if (pressed) selected.add(chip.dataset.exValue);
        else selected.delete(chip.dataset.exValue);
        apply();
        return;
      }
      if (event.target.closest('[data-ex-reset]')) reset();
    });

    if (noResEl) {
      noResEl.addEventListener('click', function (event) {
        if (event.target.closest('[data-ex-reset]')) reset();
      });
    }

    // Idempotent start: overwrites any state restored from a cached document or
    // from browser form restoration.
    reset();
    bar.hidden = false;
  }

  if (typeof document$ !== 'undefined') {
    document$.subscribe(init);
  } else {
    document.addEventListener('DOMContentLoaded', init);
  }
})();
