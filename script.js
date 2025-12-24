// Navigation functionality
let currentSection = 'home';

function showSection(sectionId) {
    // Hide all sections
    document.querySelectorAll('.page-section').forEach(section => {
        section.classList.add('hidden');
    });

    // Show target section
    document.getElementById(sectionId).classList.remove('hidden');

    // Update nav links
    document.querySelectorAll('.nav-link').forEach(link => {
        link.classList.remove('active');
    });

    // Update active nav link (handle log pages)
    const navSection = sectionId.startsWith('log-') ? 'logs' : sectionId;
    const navLink = document.querySelector(`[href="#${navSection}"]`);
    if (navLink) {
        navLink.classList.add('active');
    }

    currentSection = sectionId;
}

function showLog(logId) {
    showSection(logId);
}

// Handle navigation clicks
document.addEventListener('DOMContentLoaded', function () {
    document.querySelectorAll('.nav-link').forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const sectionId = link.getAttribute('href').substring(1);
            showSection(sectionId);
        });
    });
});

// Show/hide navigation on scroll
let lastScrollY = window.scrollY;
const nav = document.getElementById('nav');

window.addEventListener('scroll', () => {
    const currentScrollY = window.scrollY;

    if (currentScrollY > 100) {
        nav.classList.add('visible');
    } else {
        nav.classList.remove('visible');
    }

    lastScrollY = currentScrollY;
});

// Keyboard navigation
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        // Go back to logs index if on a log page
        if (currentSection.startsWith('log-')) {
            showSection('logs');
        }
    }

    // Navigate sections with arrow keys (when no input is focused)
    if (!document.activeElement.matches('input, textarea')) {
        const sections = ['home', 'experience', 'logs'];
        const currentIndex = sections.indexOf(currentSection.startsWith('log-') ? 'logs' : currentSection);

        if (e.key === 'ArrowRight' && currentIndex < sections.length - 1) {
            showSection(sections[currentIndex + 1]);
        } else if (e.key === 'ArrowLeft' && currentIndex > 0) {
            showSection(sections[currentIndex - 1]);
        }
    }
});