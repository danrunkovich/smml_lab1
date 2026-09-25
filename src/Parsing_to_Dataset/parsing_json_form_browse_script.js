(function accumulateCars() {
    const TARGET_COUNT = 700; // Целевое количество (нужно >= 250)
    const STORAGE_KEY = 'audi_a6_dataset';

    // 1. Достаем уже сохраненные машины из памяти браузера
    let savedCars = [];
    try {
        savedCars = JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]');
    } catch (e) {
        savedCars = [];
    }

    const seenUrls = new Set(savedCars.map(c => c.url));
    const newCars = [];

    // 2. Собираем машины с текущей открытой страницы
    const allLinks = Array.from(document.querySelectorAll('a[href*="/sale/"]'));

    allLinks.forEach(a => {
        let url = a.href.split('?')[0];
        if (!url.match(/\/cars\/(?:used|new)\/sale\/[^\/]+\/[^\/]+\/\d+-[a-z0-9]+\/?$/i) && 
            !url.match(/\/cars\/(?:used|new)\/sale\/.*?\/\d+/i)) {
            return;
        }
        if (seenUrls.has(url)) return;

        let card = a;
        while (card && card !== document.body) {
            if (card.innerText && card.innerText.includes('₽') && (card.innerText.includes('км') || card.innerText.includes('л.с.'))) {
                break;
            }
            card = card.parentElement;
        }

        if (!card || card === document.body) return;
        const text = card.innerText || "";
        seenUrls.add(url);

        // Парсинг параметров
        const priceMatch = text.match(/([\d\s]{5,12})\s*₽/);
        const price = priceMatch ? parseInt(priceMatch[1].replace(/\s/g, '')) : null;

        const yearMatch = text.match(/\b(19\d\d|20\d\d)\b/);
        const year = yearMatch ? parseInt(yearMatch[1]) : null;

        const kmMatch = text.match(/([\d\s]+)\s*км/);
        let mileage = 0;
        if (kmMatch) {
            let rawKm = kmMatch[1].replace(/\s/g, '');
            if (year && rawKm.startsWith(year.toString()) && rawKm.length > 5) {
                rawKm = rawKm.slice(year.toString().length);
            }
            mileage = parseInt(rawKm) || 0;
        }

        const volMatch = text.match(/(\d+\.\d+)\s*л/);
        const engineVol = volMatch ? parseFloat(volMatch[1]) : null;

        const powerMatch = text.match(/(\d+)\s*л\.с\./);
        const power = powerMatch ? parseInt(powerMatch[1]) : null;

        let fuel = "Бензин";
        if (/дизель/i.test(text)) fuel = "Дизель";
        else if (/гибрид/i.test(text)) fuel = "Гибрид";
        else if (/электро/i.test(text)) fuel = "Электро";

        let transmission = "Робот";
        if (/автомат/i.test(text)) transmission = "Автомат";
        else if (/вариатор/i.test(text)) transmission = "Вариатор";
        else if (/механика/i.test(text)) transmission = "Механика";

        let drive = "Передний";
        if (/полный|quattro|4wd|awd/i.test(text)) drive = "Полный";
        else if (/задний|rwd/i.test(text)) drive = "Задний";

        let bodyType = "Седан";
        if (/универсал|avant|wagon/i.test(text)) bodyType = "Универсал";
        else if (/хэтчбек|лифтбек/i.test(text)) bodyType = "Лифтбек";

        if (price && year) {
            newCars.push({
                url, year, price_rub: price, mileage_km: mileage,
                power_hp: power, engine_vol_l: engineVol,
                drive, transmission, fuel_type: fuel, body_type: bodyType
            });
        }
    });

    const totalCars = [...savedCars, ...newCars];
    localStorage.setItem(STORAGE_KEY, JSON.stringify(totalCars));

    console.log(`%c[УСПЕХ] На этой странице добавлено: +${newCars.length} авто. Всего в базе: ${totalCars.length} из ${TARGET_COUNT}`, "color: green; font-size: 14px; font-weight: bold;");

    // 3. Проверяем, набрали ли нужное количество
    if (totalCars.length >= TARGET_COUNT) {
        console.log(`%c[ФИНИШ] Цель достигнута! Скачиваем файл на ${totalCars.length} записей...`, "color: blue; font-size: 16px; font-weight: bold;");
        
        const blob = new Blob([JSON.stringify(totalCars, null, 2)], { type: 'application/json' });
        const downloadLink = document.createElement('a');
        downloadLink.href = URL.createObjectURL(blob);
        downloadLink.download = 'audi_a6_master_dataset.json';
        downloadLink.click();

        // Очищаем хранилище после успешного скачивания
        localStorage.removeItem(STORAGE_KEY);
    } else {
        // Определяем номер текущей страницы и переходим на следующую
        const urlObj = new URL(window.location.href);
        let currentPage = parseInt(urlObj.searchParams.get("page") || "1");
        let nextPage = currentPage + 1;
        urlObj.searchParams.set("page", nextPage);

        console.log(`%cПеренаправляем на страницу ${nextPage}... После загрузки нажмите [Вверх] и [Enter] в консоли!`, "color: orange; font-weight: bold;");
        setTimeout(() => {
            window.location.href = urlObj.toString();
        }, 1200);
    }
})();
