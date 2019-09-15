
def plot(data, tik_descriptor):
    xx = list(range(len(data.index)))
    plt.figure(figsize=(tik_descriptor.fig_width, 6))
    for res_id, title in CANDIDATE_TITLES.items():
        plt.plot(xx, 100.0 * data['res_' + res_id].values / data['total'].values, '.-', label=title)
    plt.plot(xx, 100.0 * (data['ballots_mobile'].values + data['ballots_stationar'].values) / data['total_izb'].values, '.--', label=u'Явка')
    plt.plot(xx, 100.0 * data['ballots_mobile'].values / (data['ballots_stationar'].values + data['ballots_mobile'].values), '.--', label=u'В переносных ящиках от числа голосовавших')
    plt.grid(True)
    plt.xticks(xx, list(data.index))
    plt.ylabel(u'Процент проголосовавших')
    plt.xlabel(u'Участок')
    plt.legend()
    plt.title(u'Результаты (процент от числа избирателей)')
    plt.savefig('plot.png')