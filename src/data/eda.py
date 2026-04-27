import polars as pl

def print_quantiles(df,col,describe):
    uq50=df[col].quantile(0.5) 
    uq75=df[col].quantile(0.75)
    uq90=df[col].quantile(0.90)
    uq99=df[col].quantile(0.99)
    uq100=df[col].max()
    print(f"Квантили ({describe}): {uq50:.4f}, {uq75:.4f}, {uq90:.4f}, {uq99:.4f}, {uq100:.4f}")

events=pl.read_csv("dataset/raw/events.csv")

n_users=events["visitorid"].n_unique()
n_items=events["itemid"].n_unique()


print(events.head())
print(events.describe())
print(events.shape)
print(f"уникальных пользователей: {n_users} ; Уникальных товаров: {n_items}")


eventcounts=events.group_by("event").agg(
    pl.len().alias('len'),
    pl.col("visitorid").n_unique().alias("users"),
    pl.col("itemid").n_unique().alias("items"),
)

print(eventcounts)


grpu=events.group_by("visitorid").agg(
    pl.col("event").len().alias('interaction_count'),
    (pl.col('timestamp').drop_nulls().max()-pl.col('timestamp').drop_nulls().min()).alias("active_days")/(1000*60*60*24)
)[['interaction_count','active_days']]

grpi=events.group_by("itemid").agg(
    pl.col("event").len().alias('interaction_count'),
    (pl.col('timestamp').drop_nulls().max()-pl.col('timestamp').drop_nulls().min()).alias("active_days")/(1000*60*60*24)
)[['interaction_count','active_days']]

usersmore5=grpu.filter(pl.col("interaction_count")>=5).height


print(f"Среднее кол-во взаимодействий: {grpu['interaction_count'].mean()}")
print(f"Медиана взаимодействий: {grpu['interaction_count'].median()}")
print('-----------')
print(f"Средний разброс времени: {grpu['active_days'].mean()} дня")
print(f"Медиана времени: {grpu['active_days'].median()}")
print('-----------')
print(f"Пользователей с >5 interactions: {usersmore5}")
print_quantiles(grpu,'interaction_count','пользователи')
print_quantiles(grpi,'interaction_count','товары')
print_quantiles(grpu,'active_days','время life span (дни)')
#def computeFunnelConv(df, action1, action2, col):
#    #Конверсия по событиям
#    return df.filter(pl.col('event')==action2)[col][0]/df.filter(pl.col('event')==action1)[col][0]
#
#
#funn_conv_v_a=computeFunnelConv(eventcounts,'view','addtocart','len')
#funn_conv_a_t=computeFunnelConv(eventcounts,'addtocart','transaction','len')
#funn_conv_v_t=computeFunnelConv(eventcounts,'view','transaction','len')
#              
def compute_user_conv(df,action1,action2,window):
    #конверсия пользователей
    action1users = (
    df.filter(pl.col("event")==action1)["visitorid"]
    .n_unique()
    )
    action2users=df.group_by("visitorid").agg([
        pl.col("timestamp")
        .filter(pl.col("event") == action1)
        .min()
        .alias("first_1"),

        pl.col("timestamp")
        .filter(pl.col("event") == action2)
        .min()
        .alias("first_2"),
    ]).filter(
        pl.col("first_1").is_not_null() &
        pl.col("first_2").is_not_null() &
        (pl.col("first_2") - pl.col("first_1") >= 0) &
        (pl.col("first_2") - pl.col("first_1") < window*1000*60*60*24)
    ).height
    return(action2users/action1users)

conv_v_a=compute_user_conv(events,'view','addtocart',15)
conv_a_t=compute_user_conv(events,'addtocart','transaction',15)
conv_v_t=compute_user_conv(events,'view','transaction',15)

def compute_pair_conv(df,action1,action2,window):
    #Конверсия пар (пользователь, товар)
    action1pairs=df.filter(pl.col('event')==action1)['visitorid','itemid'].n_unique()
    
    action2pairs=df.group_by(['visitorid','itemid']).agg([
        pl.col("timestamp")
        .filter(pl.col("event") == action1)
        .min()
        .alias("first_1"),

        pl.col("timestamp")
        .filter(pl.col("event") == action2)
        .min()
        .alias("first_2"),
    ]).filter(
        pl.col("first_1").is_not_null() &
        pl.col("first_2").is_not_null() &
        (pl.col("first_2") - pl.col("first_1") >= 0) &
        (pl.col("first_2") - pl.col("first_1") < window*1000*60*60*24)
    ).height
    return(action2pairs/action1pairs)

pair_conv_v_a=compute_pair_conv(events,'view','addtocart',14)
pair_conv_a_t=compute_pair_conv(events,'addtocart','transaction',14)
pair_conv_v_t=compute_pair_conv(events,'view','transaction',14)

#print(f"Конверсия funnel (v->a): {funn_conv_v_a}")
#print(f"Конверсия funnel (a->t): {funn_conv_a_t}")
#print(f"Конверсия funnel (v->t): {funn_conv_v_t}")
print("")
print(f"Конверсия user (v->a): {conv_v_a}")
print(f"Конверсия user (a->t): {conv_a_t}")
print(f"Конверсия user (v->t): {conv_v_t}")
print("")
print(f"Конверсия pair (v->a): {pair_conv_v_a}")
print(f"Конверсия pair (a->t): {pair_conv_a_t}")
print(f"Конверсия pair (v->t): {pair_conv_v_t}")
print("--------")
#фильтрация и проверка sessions
eventskcore=(pl.scan_csv("dataset/raw/events.csv")) #lazy execution
prev=-1
itera=0 # количество проходов k-core для оценки convergence
print("K-core filtering: ")
while True:
    itera+=1
    df=(eventskcore
        .join(eventskcore.group_by("visitorid")
              .len().filter(
                  pl.col("len")>=5
              ),on='visitorid',how='semi') #semi - как inner 
                                           #но возвращает только колонки из первой таблицы(проверка на входимость)
        .join(eventskcore.group_by("itemid")
              .len().filter(
                  pl.col("len")>=5
              ),on='itemid',how='semi')
        .collect()
    )
    curr=df.height
    print(
        f"{itera}: "
        f"rows={curr} "
        f"users={df['visitorid'].n_unique()} "
        f"items={df['itemid'].n_unique()}"
    )
    
    if curr==prev:
        break
    eventskcore=df.lazy()
    prev=curr
print("---------")
events=df.sort(["visitorid", "timestamp"])
events=events.with_columns(((pl.col("timestamp")-pl.col("timestamp").shift(1))/1000/60)
                           .over("visitorid").alias("gap"))
gap=30 #больше 30 минут - сессия заканчивается
sessions=events.with_columns(pl.when((pl.col("gap").is_not_null()) & (pl.col("gap")<=gap))
                             .then(0)
                             .otherwise(1)
                           .alias("new_session"))
sessions=sessions.with_columns(pl.col("new_session")
                               .cum_sum().over("visitorid").alias("session")).drop("new_session")
#print(sessions.head(10))
sessions_per_user = (
    sessions.group_by("visitorid")
    .agg(
        pl.col("session").n_unique().alias("n_sessions")
    )
)

meanses=sessions_per_user['n_sessions'].mean()
medianses=sessions_per_user['n_sessions'].median()

print(f"среднее по сессиям среди пользователей: {meanses:.4f} ; медиана: {medianses:.4f}")
print_quantiles(sessions_per_user,"n_sessions",'количество сессий')

duration=sessions.group_by(['visitorid','session']).agg(
    ((pl.col('timestamp').max()-pl.col('timestamp').min())/1000/60)
    .alias('duration')
)

print(f"Среднее по длительности: {duration['duration'].mean():.4f}")
print(f"Медиана по длительности: {duration['duration'].median():.4f}")
print_quantiles(duration,'duration','длительность')

session_stats = (
    sessions.group_by(["visitorid","session"])
    .agg(
        pl.len().alias("session_len")
    )
)
print(f"Среднее по длине сессии: {session_stats['session_len'].mean():.4f}")
print(f"Медиана по длине сессии: {session_stats['session_len'].median():.4f}")
print_quantiles(session_stats,'session_len','длина сессии')




